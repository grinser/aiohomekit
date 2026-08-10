"""RemovePairing over CoAP completes by reading M2 back.

Writing M1 alone is not the whole procedure: accessories exist that do not
apply the removal until the response is collected. These tests drive
CoAPHomeKitConnection.remove_pairing with a stubbed encryption context and
cover what the accessory can answer with -- including the responses it may not
answer with at all, since removing our own pairing ends the session.
"""

import struct

import pytest

from aiohomekit.controller.coap.connection import CoAPHomeKitConnection
from aiohomekit.controller.coap.pdu import OpCode, PDUStatus
from aiohomekit.controller.coap.structs import (
    Pdu09Accessory,
    Pdu09AccessoryContainer,
    Pdu09Characteristic,
    Pdu09CharacteristicContainer,
    Pdu09Database,
    Pdu09Service,
    Pdu09ServiceContainer,
)
from aiohomekit.exceptions import (
    AccessoryDisconnectedError,
    AuthenticationError,
    EncryptionError,
    UnknownError,
)
from aiohomekit.protocol.tlv import HAP_TLV, TLV

PAIRING_SERVICE = 0x55
PAIRINGS_CHARACTERISTIC = 0x50
PAIRINGS_IID = 18
CONTROLLER_ID = "some-controller-id"


def _database() -> Pdu09Database:
    """An accessory exposing only the Pairing service's Pairings characteristic."""
    characteristic = Pdu09Characteristic(
        type=PAIRINGS_CHARACTERISTIC,
        instance_id=PAIRINGS_IID,
        properties=0x0003,
        presentation_format=None,
        valid_range=None,
        step_value=None,
        valid_values=None,
        valid_values_range=None,
        user_descriptor=None,
    )
    service = Pdu09Service(
        type=PAIRING_SERVICE,
        instance_id=PAIRINGS_IID - 1,
        _characteristics=[Pdu09CharacteristicContainer(characteristic=characteristic)],
        properties=0,
        linked_services=None,
    )
    accessory = Pdu09Accessory(
        instance_id=1,
        _services=[Pdu09ServiceContainer(service=service)],
    )
    return Pdu09Database(_accessories=[Pdu09AccessoryContainer(accessory=accessory)])


def _m2(state=TLV.M2, error=None) -> bytes:
    """Encode an M2 the way the accessory returns it, wrapped in a HAP value TLV."""
    entries = [(TLV.kTLVType_State, state)]
    if error is not None:
        entries.append((TLV.kTLVType_Error, error))
    return bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, bytes(TLV.encode_list(entries)))]))


class FakeEncryptionContext:
    """Records writes and serves a scripted reply to the M2 read."""

    def __init__(self, read_result=None, read_error=None):
        self.writes: list[tuple[int, bytes]] = []
        self.reads: list[int] = []
        self.calls: list[tuple[OpCode, int]] = []
        self._read_result = read_result
        self._read_error = read_error

    async def post(self, opcode, iid, data, **kwargs):
        self.calls.append((opcode, iid))
        if opcode is OpCode.CHAR_WRITE:
            self.writes.append((iid, data))
            return (0, b"")
        if opcode is OpCode.CHAR_READ:
            self.reads.append(iid)
            if self._read_error is not None:
                raise self._read_error
            result = self._read_result
            # decode_pdu returns the header's body length alongside a status.
            body_len = 0 if isinstance(result, PDUStatus) else len(result or b"")
            return (body_len, result)
        raise AssertionError(f"unexpected opcode {opcode}")


def _connection(**kwargs) -> CoAPHomeKitConnection:
    conn = CoAPHomeKitConnection.__new__(CoAPHomeKitConnection)
    conn.info = _database()
    conn.enc_ctx = FakeEncryptionContext(**kwargs)
    return conn


def _decode_m1(payload: bytes) -> dict:
    inner = dict(TLV.decode_bytes(payload))[HAP_TLV.kTLVHAPParamValue]
    return dict(TLV.decode_bytes(inner))


async def test_m1_is_a_well_formed_remove_pairing_request():
    conn = _connection(read_result=_m2())

    assert await conn.remove_pairing(CONTROLLER_ID) is True

    assert len(conn.enc_ctx.writes) == 1
    iid, payload = conn.enc_ctx.writes[0]
    assert iid == PAIRINGS_IID
    m1 = _decode_m1(payload)
    assert m1[TLV.kTLVType_State] == TLV.M1
    assert m1[TLV.kTLVType_Method] == TLV.RemovePairing
    assert m1[TLV.kTLVType_Identifier] == CONTROLLER_ID.encode()


async def test_m2_is_read_back_from_the_same_characteristic():
    """The procedure is incomplete until the response is collected."""
    conn = _connection(read_result=_m2())

    await conn.remove_pairing(CONTROLLER_ID)

    assert conn.enc_ctx.reads == [PAIRINGS_IID], "M2 must be read from the pairings characteristic"


async def test_m1_is_written_before_m2_is_read():
    """Ordering is the entire content of this fix: an implementation that read
    first and wrote second would satisfy every other assertion here."""
    conn = _connection(read_result=_m2())

    await conn.remove_pairing(CONTROLLER_ID)

    assert conn.enc_ctx.calls == [
        (OpCode.CHAR_WRITE, PAIRINGS_IID),
        (OpCode.CHAR_READ, PAIRINGS_IID),
    ]


async def test_an_error_reported_in_m2_fails_the_removal():
    conn = _connection(read_result=_m2(error=TLV.kTLVError_Unknown))

    with pytest.raises(UnknownError):
        await conn.remove_pairing(CONTROLLER_ID)


async def test_an_authentication_error_in_m2_is_surfaced_as_such():
    conn = _connection(read_result=_m2(error=TLV.kTLVError_Authentication))

    with pytest.raises(AuthenticationError):
        await conn.remove_pairing(CONTROLLER_ID)


async def test_a_rejected_m1_still_fails():
    """Pre-existing behaviour: a PDU error on the write is fatal."""
    conn = _connection(read_result=_m2())

    async def rejecting_post(opcode, iid, data, **kwargs):
        return (0, PDUStatus.INVALID_REQUEST)

    conn.enc_ctx.post = rejecting_post

    with pytest.raises(UnknownError):
        await conn.remove_pairing(CONTROLLER_ID)


async def test_an_m1_rejected_for_authentication_raises_authentication_error():
    conn = _connection(read_result=_m2())

    async def rejecting_post(opcode, iid, data, **kwargs):
        return (0, PDUStatus.INSUFFICIENT_AUTHENTICATION)

    conn.enc_ctx.post = rejecting_post

    with pytest.raises(AuthenticationError):
        await conn.remove_pairing(CONTROLLER_ID)


@pytest.mark.parametrize(
    "kwargs",
    [
        # Removing our own pairing ends the session, so M2 may never arrive.
        {"read_error": AccessoryDisconnectedError("Request timeout")},
        {"read_error": EncryptionError("Decryption of PDU POST response failed")},
        # decode_pdu unpacks a 5-byte header and builds a PDUStatus outside any
        # try, so a short or garbage reply surfaces as these rather than a status.
        {"read_error": struct.error("unpack requires a buffer of 5 bytes")},
        {"read_error": ValueError("7 is not a valid PDUStatus")},
        {"read_error": TimeoutError()},
        # The accessory answered, but not with a readable response.
        {"read_result": PDUStatus.INVALID_REQUEST},
        {"read_result": b""},
        # Not a HAP value TLV at all, and a truncated TLV body
        # (TlvParseException subclasses Exception, not ValueError).
        {"read_result": b"\xff\x01\x00"},
        {"read_result": b"\x01\x10\x00"},
    ],
    ids=[
        "session-dropped",
        "decrypt-failed",
        "short-pdu",
        "bad-status-byte",
        "timeout",
        "pdu-error",
        "empty",
        "undecodable",
        "truncated-tlv",
    ],
)
async def test_an_unconfirmed_m2_is_never_reported_as_success(kwargs):
    """Only a well-formed M2 confirms a removal.

    "M1 was accepted" proves nothing: measured on an Eve Room, writing M1 and
    closing without reading M2 returns Success and leaves the pairing in place.
    That is the failure in issue #164, and reporting success on an unreadable
    M2 reproduces it exactly -- the controller discards its own credentials
    while the accessory stays paired, needing a factory reset.

    The same measurement showed M2 does arrive on that firmware, so this path
    is genuinely exceptional rather than the routine cost of removing our own
    pairing, which is what it was previously assumed to be.
    """
    conn = _connection(**kwargs)

    with pytest.raises(UnknownError, match="could not be confirmed"):
        await conn.remove_pairing(CONTROLLER_ID)
    assert conn.enc_ctx.writes, "M1 must still have been written"


@pytest.mark.parametrize(
    ("body", "case"),
    [
        # Valid outer value TLV, inner state says M1 rather than M2. A real M2
        # from an Eve Room is 01 03 06 01 02; this is that with the state byte
        # changed, i.e. the accessory answering a different step.
        (b"\x01\x03\x06\x01\x01", "state-is-m1"),
        # Decodes, but carries no state at all.
        (b"\x01\x03\x01\x01\x41", "no-state"),
    ],
    ids=["state-is-m1", "no-state"],
)
async def test_a_decodable_reply_that_is_not_m2_is_not_confirmation(body, case):
    """The reply parsed, so none of the unreadable-M2 guards fire. It still is
    not the response this procedure is waiting for, and treating it as one puts
    us back to reporting a removal we cannot show happened.
    """
    conn = _connection(read_result=body)

    with pytest.raises(UnknownError, match="could not be confirmed"):
        await conn.remove_pairing(CONTROLLER_ID)
    assert conn.enc_ctx.writes, "M1 must still have been written"
