"""Falling back from the 0x09 bulk database read to signature reads.

Some accessories do not implement opcode 0x09 and drop it silently, which also
tears down the secured session. These tests drive _read_gatt_database with a
stubbed encryption context to cover the failure paths: how the fallback is
triggered, when the accessory is remembered as not supporting 0x09, and what
happens when the signatures themselves are unusable.
"""

import asyncio
import struct

import pytest
from aiocoap.error import Error as AiocoapError

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import (
    SIGNATURE_WALK_MAX_IID,
    CoAPHomeKitConnection,
)
from aiohomekit.controller.coap.pdu import OpCode, PDUStatus
from aiohomekit.exceptions import AccessoryDisconnectedError, EncryptionError

ACCESSORY_INFORMATION = 0x3E


def _sig(char_type, svc_type, svc_iid, fmt=0x04):
    return CharacteristicTLV(
        type=char_type,
        properties=0x10,
        presentation_format=struct.pack("<BxHxxx", fmt, 0x2700),
        service_type=svc_type.to_bytes(16, "little"),
        service_instance_id=svc_iid.to_bytes(2, "little"),
    ).encode()


class FakeEncryptionContext:
    """Answers posts from a canned {iid: signature} database.

    `gatt_error` is raised for the 0x09 probe; None means 0x09 replies with an
    unsupported-PDU status instead (session stays up).
    """

    def __init__(self, signatures, gatt_error=AccessoryDisconnectedError, gatt_body=None):
        self.signatures = signatures
        self.gatt_error = gatt_error
        self.gatt_body = gatt_body
        self.probes = 0
        self.walked_iids = []

    async def post(self, opcode, iid, data, timeout=16.0, expected_statuses=()):
        if opcode is OpCode.UNK_09_READ_GATT:
            self.probes += 1
            if self.gatt_error is not None:
                raise self.gatt_error("boom")
            return (0, self.gatt_body if self.gatt_body is not None else PDUStatus.UNSUPPORTED_PDU)

        assert opcode is OpCode.CHAR_SIG_READ
        self.walked_iids.append(iid)
        if iid in self.signatures:
            body = self.signatures[iid]
            return (len(body), body)
        return (0, PDUStatus.INVALID_INSTANCE_ID)

    async def post_all(self, opcode, iids, data):
        return [PDUStatus.INVALID_REQUEST] * len(iids)


def _connection(signatures, **kwargs):
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
    conn = CoAPHomeKitConnection(owner, "::1", 5683)
    conn.enc_ctx = FakeEncryptionContext(signatures, **kwargs)
    conn._pairing_data = {"AccessoryPairingID": "AA:BB:CC:DD:EE:FF"}

    async def _fake_pair_verify(pairing_data):
        conn.reconnected = True

    conn.reconnected = False
    conn.do_pair_verify = _fake_pair_verify
    return conn


def _chars(database):
    return sorted(c.instance_id for a in database.accessories for s in a.services for c in s.characteristics)


# Accessory Information at the bottom of the iid range and a sensor above the
# minimal-enumeration bound, close enough that the walk does not give up first.
DEVICE = {
    2: _sig(0x14, ACCESSORY_INFORMATION, 1),
    3: _sig(0x20, ACCESSORY_INFORMATION, 1),
    15: _sig(0x11, 0x96, 54),
}


@pytest.mark.parametrize(
    "error",
    [AccessoryDisconnectedError, EncryptionError, asyncio.TimeoutError, AiocoapError],
)
async def test_every_probe_failure_falls_back_and_reconnects(error):
    # A dropped 0x09 surfaces as a timeout, as a decryption failure (the accessory
    # answered 404 and the reply cannot be decrypted), or as a transport teardown.
    # All of them must rebuild the database rather than propagate.
    conn = _connection(DEVICE, gatt_error=error)

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 3, 15]
    assert conn.reconnected, "the torn-down session must be re-established"
    assert conn._gatt_unsupported


async def test_probe_error_status_does_not_tear_down_the_session():
    # A PDUStatus reply means 0x09 is unsupported but the session is healthy, so
    # there is nothing to reconnect.
    conn = _connection(DEVICE, gatt_error=None)

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 3, 15]
    assert not conn.reconnected
    assert conn._gatt_unsupported


async def test_unparsable_0x09_reply_does_not_mark_it_unsupported():
    # 0x09 answered, we just could not decode it. The accessory does implement the
    # opcode, so a later read should probe it again instead of walking blindly.
    conn = _connection(DEVICE, gatt_error=None, gatt_body=b"\xde\xad\xbe\xef")

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 3, 15]
    assert not conn._gatt_unsupported


async def test_probe_is_skipped_once_the_accessory_is_known_to_drop_it():
    conn = _connection(DEVICE)

    await conn._read_gatt_database()
    assert conn.enc_ctx.probes == 1

    await conn._read_gatt_database()
    assert conn.enc_ctx.probes == 1, "0x09 must not be probed again"


async def test_undecodable_signatures_do_not_yield_an_empty_database():
    # Signatures were read but none could be parsed. Returning an empty database
    # would be cached by the controller as a valid, characteristic-less accessory.
    conn = _connection({2: b"\xff\x03\x01\x02\x03", 3: b"\xfe\x02\x09\x09"})

    with pytest.raises(AccessoryDisconnectedError):
        await conn._read_gatt_database()


async def test_walk_stops_after_a_long_run_of_gaps():
    conn = _connection(DEVICE)

    await conn._read_gatt_database()

    # Stops well before the scan limit rather than probing every possible iid.
    assert max(conn.enc_ctx.walked_iids) < SIGNATURE_WALK_MAX_IID
    assert 15 in conn.enc_ctx.walked_iids


async def test_reconnect_without_pairing_data_fails_cleanly():
    conn = _connection(DEVICE)
    conn._pairing_data = None

    with pytest.raises(AccessoryDisconnectedError):
        await conn._read_gatt_database()


class FakeCoapContext:
    def __init__(self):
        self.shutdown_at = None
        self.clock = 0

    async def shutdown(self):
        self.shutdown_at = self.clock


async def test_reconnect_waits_for_an_enumeration_in_flight():
    """A reconnect must not tear the session down mid-enumeration.

    Pairing an accessory updates its advertised endpoint and config number, which
    schedules reconnect_soon() and a fresh enumeration. A single 0x09 read is over
    too quickly to be caught by that, but a signature walk is many sequential
    requests, so the shutdown lands in the middle of it and aborts the enumeration
    that the pairing is waiting on.
    """
    conn = _connection(DEVICE)
    coap_ctx = FakeCoapContext()
    conn.enc_ctx.coap_ctx = coap_ctx

    order = []
    original_post = conn.enc_ctx.post

    async def slow_post(*args, **kwargs):
        # Yield control on every request so a concurrent reconnect can interleave.
        await asyncio.sleep(0)
        coap_ctx.clock += 1
        return await original_post(*args, **kwargs)

    conn.enc_ctx.post = slow_post

    async def enumerate():
        result = await conn.get_accessory_info()
        order.append("enumeration finished")
        return result

    enumeration = asyncio.create_task(enumerate())
    await asyncio.sleep(0)  # let the walk get under way

    async def reconnect():
        await conn.reconnect_soon()
        order.append("reconnect finished")

    await asyncio.gather(enumeration, asyncio.create_task(reconnect()))

    assert order == ["enumeration finished", "reconnect finished"]
    assert coap_ctx.shutdown_at is not None, "the reconnect must still happen"
    assert conn.info is not None
    assert _chars(conn.info) == [2, 3, 15], "the enumeration must not be truncated"


async def test_concurrent_enumerations_are_serialised():
    """A fresh pairing runs two enumerations at once: the one the caller awaits and
    one scheduled by the config-number change. They must not interleave on the
    shared connection."""
    conn = _connection(DEVICE)
    in_flight = 0
    original_post = conn.enc_ctx.post

    async def counting_post(*args, **kwargs):
        assert in_flight == 1, "two enumerations ran against the connection at once"
        await asyncio.sleep(0)
        return await original_post(*args, **kwargs)

    conn.enc_ctx.post = counting_post

    async def enumerate():
        nonlocal in_flight
        async with conn._enumeration_lock:
            in_flight += 1
            try:
                await conn._get_accessory_info()
            finally:
                in_flight -= 1

    await asyncio.gather(enumerate(), enumerate())

    assert _chars(conn.info) == [2, 3, 15]
