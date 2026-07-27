"""Falling back from the 0x09 bulk database read to signature reads.

Some accessories do not implement opcode 0x09 and drop it silently, which also
tears down the secured session. These tests drive _read_gatt_database with a
stubbed encryption context to cover the failure paths: how the fallback is
triggered, when the accessory is remembered as not supporting 0x09, when the
rebuilt database is only part of the accessory, and what happens when the
signatures themselves are unusable.
"""

import asyncio
import struct

import pytest
from aiocoap.error import Error as AiocoapError

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import (
    _WALK_EXPECTED_STATUSES,
    DEFAULT_POST_TIMEOUT,
    GATT_PROBE_TIMEOUT,
    SIGNATURE_WALK_MAX_IID,
    SIGNATURE_WALK_MAX_MISSES,
    CoAPHomeKitConnection,
)
from aiohomekit.controller.coap.pdu import OpCode, PDUStatus
from aiohomekit.exceptions import AccessoryDisconnectedError, EncryptionError
from aiohomekit.protocol.tlv import HAP_TLV, TLV

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

    Emulates the one behaviour the fallback turns on: firmware that drops 0x09
    also drops the session, so a probe failure clears coap_ctx exactly as
    post_bytes would, and the connection is not usable again until pair-verify
    restores it.
    """

    def __init__(
        self,
        signatures,
        gatt_error=AccessoryDisconnectedError,
        gatt_body=None,
        gatt_status=PDUStatus.UNSUPPORTED_PDU,
        values=None,
    ):
        self.signatures = signatures
        self.gatt_error = gatt_error
        self.gatt_body = gatt_body
        self.gatt_status = gatt_status
        self.values = values
        self.coap_ctx = object()
        self.probes = 0
        self.walked_iids = []
        self.probe_timeouts = []
        self.walk_expected_statuses = []

    async def post(self, opcode, iid, data, timeout=16.0, expected_statuses=()):
        if opcode is OpCode.UNK_09_READ_GATT:
            self.probes += 1
            self.probe_timeouts.append(timeout)
            if self.gatt_error is not None:
                # A dropped 0x09 takes the session with it.
                self.coap_ctx = None
                raise self.gatt_error("boom")
            if self.gatt_body is not None:
                return (len(self.gatt_body), self.gatt_body)
            return (0, self.gatt_status)

        assert opcode is OpCode.CHAR_SIG_READ
        self.walk_expected_statuses.append(expected_statuses)
        self.walked_iids.append(iid)
        if iid in self.signatures:
            body = self.signatures[iid]
            return (len(body), body)
        return (0, PDUStatus.INVALID_INSTANCE_ID)

    async def post_all(self, opcode, iids, data):
        if self.values is None:
            return [PDUStatus.INVALID_REQUEST] * len(iids)
        return [self.values.get(iid, PDUStatus.INVALID_REQUEST) for iid in iids]


def _connection(signatures, **kwargs):
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
    conn = CoAPHomeKitConnection(owner, "::1", 5683)
    conn.enc_ctx = FakeEncryptionContext(signatures, **kwargs)
    conn._pairing_data = {"AccessoryPairingID": "AA:BB:CC:DD:EE:FF"}

    async def _fake_pair_verify(pairing_data):
        conn.reconnected = True
        conn.enc_ctx.coap_ctx = object()

    conn.reconnected = False
    conn.do_pair_verify = _fake_pair_verify
    return conn


def _chars(database):
    return sorted(c.instance_id for a in database.accessories for s in a.services for c in s.characteristics)


def _value(raw: bytes) -> bytes:
    # bytes, not bytearray: real PDU bodies are slices of the decrypted response.
    return bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, raw)]))


# Accessory Information at the bottom of the iid range and a sensor above the
# minimal-enumeration bound, close enough that the walk does not give up first.
DEVICE = {
    2: _sig(0x14, ACCESSORY_INFORMATION, 1),
    3: _sig(0x20, ACCESSORY_INFORMATION, 1),
    15: _sig(0x11, 0x96, 54),
}


@pytest.mark.parametrize(
    "error",
    [
        AccessoryDisconnectedError,
        EncryptionError,
        TimeoutError,
        AiocoapError,
        # decode_pdu unpacks a header and builds a PDUStatus outside any try, so
        # a short or garbage reply surfaces as one of these.
        struct.error,
        ValueError,
    ],
)
async def test_every_probe_failure_falls_back_and_reconnects(error):
    conn = _connection(DEVICE, gatt_error=error)

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 3, 15]
    assert conn.reconnected, "the dropped 0x09 tore the session down; it must be re-established"
    assert conn._gatt_unsupported


async def test_probe_error_status_does_not_tear_down_the_session():
    conn = _connection(DEVICE, gatt_error=None)

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 3, 15]
    assert not conn.reconnected, "a status reply means the session is still healthy"


async def test_unparsable_0x09_reply_does_not_mark_it_unsupported():
    conn = _connection(DEVICE, gatt_error=None, gatt_body=b"\xde\xad\xbe\xef")

    await conn._read_gatt_database()

    assert not conn._gatt_unsupported, "0x09 did answer, so it must not be latched off"


@pytest.mark.parametrize(
    "status",
    [PDUStatus.UNSUPPORTED_PDU, PDUStatus.INVALID_REQUEST, PDUStatus.INVALID_INSTANCE_ID],
)
async def test_a_definitive_rejection_latches_0x09_off(status):
    conn = _connection(DEVICE, gatt_error=None, gatt_status=status)

    await conn._read_gatt_database()

    assert conn._gatt_unsupported


@pytest.mark.parametrize(
    "status",
    [
        PDUStatus.MAX_PROCEDURES,
        PDUStatus.INSUFFICIENT_AUTHENTICATION,
        PDUStatus.INSUFFICIENT_AUTHORIZATION,
        PDUStatus.TID_MISMATCH,
        PDUStatus.BAD_CONTROL,
    ],
)
async def test_a_transient_status_does_not_latch_0x09_off(status):
    """Busy or desynced is not "unsupported".

    The flag is never cleared, so latching on a transient hiccup would downgrade
    a perfectly capable accessory to a 300-request walk for good.
    """
    conn = _connection(DEVICE, gatt_error=None, gatt_status=status)

    await conn._read_gatt_database()

    assert not conn._gatt_unsupported


async def test_0x09_is_latched_off_even_if_the_walk_then_fails():
    """Latching must not depend on the walk succeeding.

    Otherwise one timed-out signature read leaves the flag clear and the next
    connect re-pays the probe -- and the session teardown it causes -- forever.
    """
    conn = _connection({}, gatt_error=AccessoryDisconnectedError)

    async def failing_walk(max_iid=SIGNATURE_WALK_MAX_IID):
        raise AccessoryDisconnectedError("walk died")

    conn._signature_walk = failing_walk

    with pytest.raises(AccessoryDisconnectedError):
        await conn._read_gatt_database()

    assert conn._gatt_unsupported


async def test_probe_is_skipped_once_the_accessory_is_known_to_drop_it():
    conn = _connection(DEVICE)

    await conn._read_gatt_database()
    probes_after_first = conn.enc_ctx.probes
    await conn._read_gatt_database()

    assert conn.enc_ctx.probes == probes_after_first == 1


async def test_undecodable_signatures_do_not_yield_an_empty_database():
    conn = _connection({2: b"\x00\x01\x02"})

    with pytest.raises(AccessoryDisconnectedError, match="no characteristic signature"):
        await conn._read_gatt_database()


async def test_signatures_without_a_service_are_discarded():
    """Defaulting to service 0 would collapse them into one synthetic service."""
    orphan = CharacteristicTLV(type=0x14, properties=0x10).encode()
    conn = _connection({2: orphan})

    with pytest.raises(AccessoryDisconnectedError, match="no characteristic signature"):
        await conn._read_gatt_database()


async def test_walk_stops_after_a_long_run_of_gaps():
    """Pins the stop point exactly, so removing the miss counter fails here."""
    conn = _connection(DEVICE)

    await conn._read_gatt_database()

    last_hit = max(DEVICE)
    assert conn.enc_ctx.walked_iids == list(range(1, last_hit + SIGNATURE_WALK_MAX_MISSES + 1))


async def test_full_enumeration_is_not_reported_as_partial():
    conn = _connection(DEVICE)

    await conn._read_gatt_database()

    assert not conn.database_is_partial


async def test_a_full_walk_that_runs_out_of_range_is_reported_as_partial():
    """Truncation is about what the walk did, not what bound was requested.

    An accessory numbered densely enough never trips the miss counter, so the
    walk ends at the scan limit with the database possibly incomplete.
    """
    dense = {iid: _sig(0x14, ACCESSORY_INFORMATION, 1) for iid in range(2, SIGNATURE_WALK_MAX_IID, 10)}
    conn = _connection(dense)

    await conn._read_gatt_database()

    assert conn.enc_ctx.walked_iids[-1] == SIGNATURE_WALK_MAX_IID
    assert conn.database_is_partial


async def test_a_transient_walk_failure_aborts_instead_of_truncating():
    """A busy accessory must not be mistaken for the end of the database."""
    conn = _connection(DEVICE)
    original_post = conn.enc_ctx.post

    async def flaky_post(opcode, iid, data, **kwargs):
        if opcode is OpCode.CHAR_SIG_READ and iid == 4:
            return (0, PDUStatus.MAX_PROCEDURES)
        return await original_post(opcode, iid, data, **kwargs)

    conn.enc_ctx.post = flaky_post

    with pytest.raises(AccessoryDisconnectedError, match="iid 4"):
        await conn._read_gatt_database()


async def test_walk_stops_when_the_session_dies_under_it():
    conn = _connection(DEVICE)
    original_post = conn.enc_ctx.post

    async def dying_post(opcode, iid, data, **kwargs):
        if opcode is OpCode.CHAR_SIG_READ and iid == 5:
            conn.enc_ctx.coap_ctx = None
        return await original_post(opcode, iid, data, **kwargs)

    conn.enc_ctx.post = dying_post

    with pytest.raises(AccessoryDisconnectedError, match="Session ended"):
        await conn._read_gatt_database()


async def test_reconnect_without_pairing_data_fails_cleanly():
    conn = _connection(DEVICE)
    conn._pairing_data = None

    with pytest.raises(AccessoryDisconnectedError, match="pairing data unavailable"):
        await conn._read_gatt_database()


async def test_the_probe_and_the_walk_declare_their_expectations():
    """Without these the probe would use the default timeout and the walk would
    log a warning for every gap -- 306 of them on the accessory this targets."""
    conn = _connection(DEVICE, gatt_error=None)

    await conn._read_gatt_database()

    assert conn.enc_ctx.probe_timeouts == [GATT_PROBE_TIMEOUT]
    # A probe that times out latches 0x09 off for the life of the connection, so
    # it must not be stricter than the default every other request gets: an
    # accessory that answers a large database slowly is supported, not broken.
    assert GATT_PROBE_TIMEOUT >= DEFAULT_POST_TIMEOUT
    assert conn.enc_ctx.walk_expected_statuses
    assert all(e == _WALK_EXPECTED_STATUSES for e in conn.enc_ctx.walk_expected_statuses)


async def test_values_are_read_onto_a_rebuilt_database():
    """A uint32 value only round-trips if the rebuilt characteristic kept the
    presentation format from its signature; truncation here would mean the
    walk lost it."""
    device = dict(DEVICE)
    device[4] = _sig(0x21, ACCESSORY_INFORMATION, 1, fmt=0x08)
    conn = _connection(device, values={4: _value(struct.pack("<I", 70000))})

    await conn.get_accessory_info()

    characteristic = conn.info.find_characteristic_by_iid(4)
    assert characteristic.value == 70000, "a uint32 must not be truncated to uint8"
    assert _chars(conn.info) == [2, 3, 4, 15]


async def test_value_read_failures_do_not_discard_the_database():
    conn = _connection(DEVICE)

    await conn.get_accessory_info()

    assert _chars(conn.info) == [2, 3, 15], "the database survives unreadable values"


async def test_re_verify_is_serialised_so_only_one_session_is_established():
    """connect() and a fallback re-verify both run pair-verify and both assign
    enc_ctx. Unserialised, the loser's session is never referenced again --
    leaking a socket and a session slot on an accessory that has few."""
    conn = _connection(DEVICE)
    conn.enc_ctx.coap_ctx = None
    verifies = 0

    async def slow_verify(pairing_data):
        nonlocal verifies
        verifies += 1
        await asyncio.sleep(0)
        conn.enc_ctx.coap_ctx = object()

    conn.do_pair_verify = slow_verify

    await asyncio.gather(conn._reverify_session(), conn._reverify_session())

    assert verifies == 1, "the second caller must see the session already restored"


async def test_concurrent_enumerations_are_serialised():
    """Config-entry setup and a config-changed notification can both enumerate.

    Unserialised, the second replaces self.info while the first is still writing
    values into the old objects, and the first publishes a valueless database.
    Note this drives the public API and takes no lock of its own -- a test that
    supplied the mutual exclusion itself would pass with the lock removed.
    """
    conn = _connection(DEVICE)
    in_flight = 0
    peak = 0
    original_post = conn.enc_ctx.post

    async def counting_post(opcode, iid, data, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        try:
            return await original_post(opcode, iid, data, **kwargs)
        finally:
            in_flight -= 1

    conn.enc_ctx.post = counting_post

    await asyncio.gather(conn.get_accessory_info(), conn.get_accessory_info())

    assert peak == 1, "two enumerations overlapped on the same connection"
    assert _chars(conn.info) == [2, 3, 15]


async def test_a_protected_characteristic_is_skipped_not_fatal():
    """An authorization refusal describes one characteristic, not the session."""
    conn = _connection(DEVICE)
    original_post = conn.enc_ctx.post

    async def protected_post(opcode, iid, data, **kwargs):
        if opcode is OpCode.CHAR_SIG_READ and iid == 3:
            return (0, PDUStatus.INSUFFICIENT_AUTHENTICATION)
        return await original_post(opcode, iid, data, **kwargs)

    conn.enc_ctx.post = protected_post

    database = await conn._read_gatt_database()

    assert _chars(database) == [2, 15], "the rest of the database must survive"


async def test_connect_does_not_verify_again_over_a_restored_session():
    """connect() checks is_connected before taking the verify lock, so it must
    re-check under it: otherwise it shuts down the session another task just
    established and replaces it, and that task's next request fails."""
    # 0x09 answers here, so the enumeration inside connect() does not itself
    # re-verify: the only pair-verifies are the two racing ones.
    conn = _connection(DEVICE, gatt_error=None)
    conn.enc_ctx.coap_ctx = None
    verifies = 0

    async def slow_verify(pairing_data):
        nonlocal verifies
        verifies += 1
        await asyncio.sleep(0)
        conn.enc_ctx.coap_ctx = object()

    conn.do_pair_verify = slow_verify

    await asyncio.gather(conn._reverify_session(), conn.connect({"AccessoryPairingID": "x"}))

    assert verifies == 1, "the session was already up by the time connect() got the lock"
