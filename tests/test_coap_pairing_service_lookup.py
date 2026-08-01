"""Pairing operations on a connection that has not enumerated the accessory.

list_pairings and remove_pairing need exactly one characteristic. On an
accessory that drops 0x09, obtaining it via the full signature walk takes
longer than a controller will wait to remove a pairing, and giving up leaves
the pairing orphaned on the accessory. The lookup therefore escalates: use the
database at hand, then a read bounded to PAIRING_SERVICE_MAX_IID, then a full
walk -- and whatever a bounded read builds is never published as complete.
"""

import asyncio
import struct

import pytest

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import (
    GATT_PROBE_TIMEOUT,
    PAIRING_PROBE_TIMEOUT,
    PAIRING_SERVICE_MAX_IID,
    SIGNATURE_WALK_MAX_IID,
    CoAPHomeKitConnection,
)
from aiohomekit.controller.coap.pdu import OpCode, PDUStatus
from aiohomekit.exceptions import AccessoryDisconnectedError, UnknownError
from aiohomekit.protocol.tlv import HAP_TLV, TLV

ACCESSORY_INFORMATION = 0x3E
PAIRING_SERVICE = 0x55
PAIRINGS_CHARACTERISTIC = 0x50


def _sig(char_type, svc_type, svc_iid, fmt=0x04):
    return CharacteristicTLV(
        type=char_type,
        properties=0x10,
        presentation_format=struct.pack("<BxHxxx", fmt, 0x2700),
        service_type=svc_type.to_bytes(16, "little"),
        service_instance_id=svc_iid.to_bytes(2, "little"),
    ).encode()


def _m2() -> bytes:
    inner = bytes(TLV.encode_list([(TLV.kTLVType_State, TLV.M2)]))
    return bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, inner)]))


class FakeEncryptionContext:
    """A 0x09-dropping accessory that answers signature walks and the
    RemovePairing write/read exchange."""

    def __init__(self, signatures):
        self.signatures = signatures
        self.coap_ctx = object()
        self.walked = []
        self.writes = []
        self.reads = []
        self.probe_timeouts = []

    async def post(self, opcode, iid, data, timeout=16.0, expected_statuses=()):
        if opcode is OpCode.UNK_09_READ_GATT:
            self.probe_timeouts.append(timeout)
            # Dropped, and it takes the session with it.
            self.coap_ctx = None
            raise AccessoryDisconnectedError("no reply")
        if opcode is OpCode.CHAR_SIG_READ:
            self.walked.append(iid)
            if iid in self.signatures:
                body = self.signatures[iid]
                return (len(body), body)
            return (0, PDUStatus.INVALID_INSTANCE_ID)
        if opcode is OpCode.CHAR_WRITE:
            self.writes.append(iid)
            return (0, b"")
        if opcode is OpCode.CHAR_READ:
            self.reads.append(iid)
            m2 = _m2()
            return (len(m2), m2)
        raise AssertionError(f"unexpected opcode {opcode}")

    async def post_all(self, opcode, iids, data):
        return [PDUStatus.INVALID_REQUEST] * len(iids)


def _connection(signatures):
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
    conn = CoAPHomeKitConnection(owner, "::1", 5683)
    conn.enc_ctx = FakeEncryptionContext(signatures)
    conn._pairing_data = {"AccessoryPairingID": "AA:BB:CC:DD:EE:FF"}

    async def _fake_pair_verify(pairing_data):
        conn.enc_ctx.coap_ctx = object()

    conn.do_pair_verify = _fake_pair_verify
    return conn


# The Pairings characteristic within the bounded range.
DEVICE = {
    2: _sig(0x14, ACCESSORY_INFORMATION, 1),
    3: _sig(0x20, ACCESSORY_INFORMATION, 1),
    18: _sig(PAIRINGS_CHARACTERISTIC, PAIRING_SERVICE, 17),
}


async def test_remove_pairing_on_a_cold_connection_stays_within_the_bound():
    """The whole point: a controller will not wait out a 300-iid walk to remove
    a pairing, so the first fallback read must stop at the bound."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.writes == [18], "M1 must go to the Pairings characteristic"
    assert max(conn.enc_ctx.walked) == PAIRING_SERVICE_MAX_IID, "the read must stop at the bound"


async def test_the_lookup_escalates_to_a_full_walk_when_the_bound_missed():
    device = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),
        # Anchors keep every gap under the miss counter so the walk continues.
        20: _sig(0x11, 0x96, 19),
        40: _sig(PAIRINGS_CHARACTERISTIC, PAIRING_SERVICE, 39),
    }
    conn = _connection(device)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.writes == [40]
    assert conn.enc_ctx.walked.count(1) == 2, "one bounded read, then one full walk"
    assert max(conn.enc_ctx.walked) > PAIRING_SERVICE_MAX_IID


async def test_the_lookup_gives_up_cleanly_when_there_is_no_pairing_service():
    conn = _connection({2: _sig(0x14, ACCESSORY_INFORMATION, 1)})

    with pytest.raises(UnknownError, match="no Pairing service"):
        await conn.remove_pairing("controller-id")

    assert not conn.enc_ctx.writes, "nothing may be written without the characteristic"


async def test_a_bounded_read_publishes_no_database_at_all():
    """PAIRING_SERVICE_MAX_IID exceeds SIGNATURE_WALK_MAX_MISSES, so a bounded
    read can terminate on the miss counter and look 'complete'. It is not: the
    bound was chosen to find one characteristic. Rather than marking the result
    partial and relying on every later reader to honour that, the lookup keeps
    it to itself -- there is no way to misuse a database that was never
    published."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.info is None, "the bounded read published its truncated database"
    assert not conn.database_from_walk, "the bounded read claimed to have enumerated"


async def test_the_pairings_lookup_does_not_wait_for_a_running_enumeration():
    """An unpair is on a deadline the controller enforces: on hardware a full
    walk ran to iid 84 and the unpair was cancelled 0.3 s before it finished,
    which orphans the pairing on the accessory. So the lookup must not queue
    behind an enumeration somebody else started -- which is safe precisely
    because it publishes nothing (see above) and so has nothing to clobber."""
    conn = _connection(DEVICE)
    walk_started = asyncio.Event()
    hold_the_walk = asyncio.Event()
    original_walk = conn._signature_walk

    async def blocked_full_walk(max_iid=SIGNATURE_WALK_MAX_IID):
        if max_iid == SIGNATURE_WALK_MAX_IID:
            walk_started.set()
            await hold_the_walk.wait()
        return await original_walk(max_iid)

    conn._signature_walk = blocked_full_walk

    full = asyncio.create_task(conn.get_accessory_info())
    await walk_started.wait()

    removal = asyncio.create_task(conn.remove_pairing("controller-id"))
    try:
        await asyncio.wait_for(asyncio.shield(removal), timeout=0.25)
        blocked = False
    except asyncio.TimeoutError:
        blocked = True
    finally:
        hold_the_walk.set()
        for task in (full, removal):
            task.cancel()
        await asyncio.gather(full, removal, return_exceptions=True)

    assert not blocked, "the unpair queued behind a full walk it did not start"
    assert conn.enc_ctx.writes == [18]


async def test_connect_does_not_enumerate_for_a_pairing_operation():
    """The regression that made pr-e useless in the field: remove_pairing goes
    through _ensure_connected, and connect() used to run a full
    get_accessory_info() before the bounded lookup ever got a turn. Measured on
    hardware, that full walk ran to iid 84 and the controller cancelled the
    unpair 0.3 s before it finished."""
    conn = _connection(DEVICE)
    conn.enc_ctx.coap_ctx = None  # else connect() returns before doing anything
    enumerations = []

    original = conn.get_accessory_info

    async def record(verify_attempts=1):
        enumerations.append(verify_attempts)
        return await original(verify_attempts)

    conn.get_accessory_info = record

    await conn.connect({"AccessoryPairingID": "x"}, enumerate_database=False)

    assert enumerations == [], "a pairing connect must not enumerate the accessory"


async def test_connect_still_enumerates_for_everything_else():
    conn = _connection(DEVICE)
    conn.enc_ctx.coap_ctx = None  # else connect() returns before doing anything
    enumerations = []
    original = conn.get_accessory_info

    async def record(verify_attempts=1):
        enumerations.append(verify_attempts)
        return await original(verify_attempts)

    conn.get_accessory_info = record

    await conn.connect({"AccessoryPairingID": "x"}, attempts=3)

    assert enumerations == [3], "polling still needs the database, on the caller's budget"


async def test_a_bounded_read_uses_the_short_0x09_probe():
    """On firmware that drops 0x09 the probe is pure latency, and a pairing
    operation is already being timed by the controller -- 20 s of it was half
    the measured unpair budget."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.probe_timeouts == [PAIRING_PROBE_TIMEOUT]
    assert PAIRING_PROBE_TIMEOUT < GATT_PROBE_TIMEOUT


async def test_a_full_read_keeps_the_generous_probe():
    conn = _connection(DEVICE)

    await conn.get_accessory_info()

    assert conn.enc_ctx.probe_timeouts == [GATT_PROBE_TIMEOUT]
