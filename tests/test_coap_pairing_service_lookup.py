"""Pairing operations on a connection with no cached accessory database.

list_pairings and remove_pairing need exactly one characteristic: the Pairing
service's Pairings characteristic. Normally its instance id comes from the
controller's cache without touching the network -- see the INV-17 tests, which
cover the warm case, and which is the only case that occurs in the field
because a controller cannot delete a config entry that never had an entity map.

This file covers the cold path: no cache, so the iid has to come from the
accessory. That path is allowed to be slow. It is not allowed to be wrong,
because if the fast path is the only one that works then a cache miss silently
orphans the device.

A previous design bounded the walk to a fixed iid to make the cold path fit a
controller's patience. It did not fit -- 32 sequential reads is ~16 s on real
hardware against a window measured as low as 10 s -- and it has been deleted
rather than kept as a fallback that cannot work. A cold unpair may now queue
behind an enumeration; there is nothing else it could do, since without a cache
the accessory is the only source for the iid.
"""

import asyncio
import struct

import pytest

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import (
    GATT_PROBE_TIMEOUT,
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
    """A connection whose owner has no cached accessories -- the cold path.

    `accessories = None` is deliberate here and load-bearing: it is what makes
    this file exercise the network route. Elsewhere the same construction was
    an accident that hid a defect for days, so it is spelled out.
    """
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
    conn = CoAPHomeKitConnection(owner, "::1", 5683)
    conn.enc_ctx = FakeEncryptionContext(signatures)
    conn._pairing_data = {"AccessoryPairingID": "AA:BB:CC:DD:EE:FF"}

    async def _fake_pair_verify(pairing_data):
        conn.enc_ctx.coap_ctx = object()

    conn.do_pair_verify = _fake_pair_verify
    return conn


DEVICE = {
    2: _sig(0x14, ACCESSORY_INFORMATION, 1),
    3: _sig(0x20, ACCESSORY_INFORMATION, 1),
    18: _sig(PAIRINGS_CHARACTERISTIC, PAIRING_SERVICE, 17),
}


async def test_a_cold_remove_pairing_still_removes_the_pairing():
    """The fallback has to work, or a cache miss orphans the device."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.writes == [18], "M1 must go to the Pairings characteristic"


async def test_a_cold_lookup_finds_a_characteristic_beyond_any_former_bound():
    """The deleted design stopped at iid 32 and missed accessories that number
    their services further out. The full enumeration has no such horizon."""
    device = {
        2: _sig(0x14, ACCESSORY_INFORMATION, 1),
        # Anchors keep every gap under the miss counter so the walk continues.
        20: _sig(0x11, 0x96, 19),
        40: _sig(PAIRINGS_CHARACTERISTIC, PAIRING_SERVICE, 39),
    }
    conn = _connection(device)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.writes == [40]
    assert max(conn.enc_ctx.walked) > 32


async def test_the_lookup_gives_up_cleanly_when_there_is_no_pairing_service():
    conn = _connection({2: _sig(0x14, ACCESSORY_INFORMATION, 1)})

    with pytest.raises(UnknownError, match="no Pairing service"):
        await conn.remove_pairing("controller-id")

    assert not conn.enc_ctx.writes, "nothing may be written without the characteristic"


async def test_a_cold_lookup_publishes_a_complete_database():
    """The cold path goes through the ordinary enumeration, so what it installs
    is a real database -- not the truncated one the bounded read used to build
    and have to be prevented from publishing."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.info is not None
    assert not conn.database_is_partial, "an enumeration that ran to completion is not partial"


async def test_connect_does_not_enumerate_for_a_pairing_operation():
    """remove_pairing goes through _ensure_connected, and connect() used to run
    a full get_accessory_info() before the lookup got a turn. Measured on
    hardware, that walk ran to iid 84 and the controller cancelled the unpair
    0.3 s before it finished."""
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
    conn.enc_ctx.coap_ctx = None
    enumerations = []
    original = conn.get_accessory_info

    async def record(verify_attempts=1):
        enumerations.append(verify_attempts)
        return await original(verify_attempts)

    conn.get_accessory_info = record

    await conn.connect({"AccessoryPairingID": "x"}, attempts=3)

    assert enumerations == [3], "polling still needs the database, on the caller's budget"


async def test_the_cold_path_keeps_the_generous_probe():
    """No caller may shorten the 0x09 probe: failing it latches the accessory
    onto the walk for the life of the connection."""
    conn = _connection(DEVICE)

    assert await conn.remove_pairing("controller-id") is True

    assert conn.enc_ctx.probe_timeouts == [GATT_PROBE_TIMEOUT]


async def test_a_cold_unpair_may_queue_behind_an_enumeration():
    """Documented, not desired. Without a cache the accessory is the only
    source for the iid, so there is nothing to do but wait for the enumeration
    lock. The warm path -- the one that occurs in the field -- must not queue;
    that is asserted in test_coap_invariants.py.
    """
    conn = _connection(DEVICE)
    walk_started = asyncio.Event()
    hold_the_walk = asyncio.Event()
    original_walk = conn._signature_walk

    async def blocked_walk(max_iid=300):
        walk_started.set()
        await hold_the_walk.wait()
        return await original_walk(max_iid)

    conn._signature_walk = blocked_walk

    full = asyncio.create_task(conn.get_accessory_info())
    await walk_started.wait()
    removal = asyncio.create_task(conn.remove_pairing("controller-id"))
    try:
        await asyncio.wait_for(asyncio.shield(removal), timeout=0.1)
        queued = False
    except asyncio.TimeoutError:
        queued = True
    finally:
        hold_the_walk.set()
        for task in (full, removal):
            task.cancel()
        await asyncio.gather(full, removal, return_exceptions=True)

    assert queued, (
        "a cold unpair completed without the enumeration lock; if the cold path "
        "has become cheap, delete this test rather than relaxing it"
    )
