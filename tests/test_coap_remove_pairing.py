"""RemovePairing over CoAP completes by reading M2 back."""

import asyncio
import struct

import pytest
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

import aiohomekit.controller.coap.connection as coap_connection
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
    InvalidError,
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
        self.write_kwargs: dict = {}
        self.read_kwargs: dict = {}
        self._read_result = read_result
        self._read_error = read_error

    async def post(self, opcode, iid, data, **kwargs):
        self.calls.append((opcode, iid))
        if opcode is OpCode.CHAR_WRITE:
            self.writes.append((iid, data))
            self.write_kwargs = kwargs
            return (0, b"")
        if opcode is OpCode.CHAR_READ:
            self.reads.append(iid)
            self.read_kwargs = kwargs
            if self._read_error is not None:
                raise self._read_error
            result = self._read_result
            # decode_pdu returns the header's body length alongside a status.
            body_len = 0 if isinstance(result, PDUStatus) else len(result or b"")
            return (body_len, result)
        raise AssertionError(f"unexpected opcode {opcode}")


def _connection(**kwargs) -> CoAPHomeKitConnection:
    conn = CoAPHomeKitConnection.__new__(CoAPHomeKitConnection)
    # No owner: the Pairings iid must come from self.info, as it does for a
    # pairing object built by hand.
    conn.owner = None
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
    """Ordering is the whole fix: reading first and writing second would satisfy every other assertion."""
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
        # Removing our own pairing may end the session before M2 arrives.
        {"read_error": AccessoryDisconnectedError("Request timeout")},
        {"read_error": EncryptionError("Decryption of PDU POST response failed")},
        # decode_pdu unpacks a 5-byte header and builds a PDUStatus outside any
        # try, so a short or garbage reply surfaces as these rather than a status.
        {"read_error": struct.error("unpack requires a buffer of 5 bytes")},
        {"read_error": ValueError("7 is not a valid PDUStatus")},
        {"read_error": TimeoutError()},
        # The accessory answered, but not with a readable response.
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
        "empty",
        "undecodable",
        "truncated-tlv",
    ],
)
async def test_an_unconfirmed_m2_is_never_reported_as_success(kwargs):
    """An M2 that cannot be read leaves the removal unconfirmed: AccessoryDisconnectedError, never success."""
    conn = _connection(**kwargs)

    with pytest.raises(AccessoryDisconnectedError, match="could not be confirmed"):
        await conn.remove_pairing(CONTROLLER_ID)
    assert conn.enc_ctx.writes, "M1 must still have been written"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (PDUStatus.INVALID_REQUEST, UnknownError),
        (PDUStatus.INSUFFICIENT_AUTHENTICATION, AuthenticationError),
        (PDUStatus.INSUFFICIENT_AUTHORIZATION, AuthenticationError),
    ],
    ids=["invalid-request", "insufficient-authentication", "insufficient-authorization"],
)
async def test_a_rejected_m2_read_fails_like_a_rejected_m1(status, expected):
    """A PDU error status on the read is mapped like one on the M1 write."""
    conn = _connection(read_result=status)

    with pytest.raises(expected, match="Remove pairing failed"):
        await conn.remove_pairing(CONTROLLER_ID)


async def test_a_reply_with_the_wrong_state_is_not_confirmation():
    """A reply with the wrong state raises InvalidError, as on the BLE and IP transports."""
    conn = _connection(read_result=b"\x01\x03\x06\x01\x01")

    with pytest.raises(InvalidError):
        await conn.remove_pairing(CONTROLLER_ID)


async def test_a_reply_without_a_state_field_is_accepted():
    """A missing state field defaults to M2, as on the BLE and IP transports."""
    conn = _connection(read_result=b"\x01\x03\x01\x01\x41")

    assert await conn.remove_pairing(CONTROLLER_ID) is True


async def test_the_m2_read_carries_its_own_timeout():
    """The read is bounded by REMOVE_PAIRING_M2_TIMEOUT, passed through post()."""
    conn = _connection(read_result=_m2())

    await conn.remove_pairing(CONTROLLER_ID)

    assert conn.enc_ctx.read_kwargs == {"timeout": coap_connection.REMOVE_PAIRING_M2_TIMEOUT}
    assert "timeout" not in conn.enc_ctx.write_kwargs


async def test_a_read_that_never_returns_times_out_and_ends_the_session():
    """On a real EncryptionContext an unanswered request raises within the timeout and shuts the CoAP context down."""

    class HangingRequest:
        @property
        def response(self):
            return asyncio.Event().wait()

    class HangingCoapContext:
        shutdowns = 0

        def request(self, message):
            return HangingRequest()

        async def shutdown(self):
            self.shutdowns += 1

    key = ChaCha20Poly1305(bytes(32))
    coap_ctx = HangingCoapContext()
    ctx = coap_connection.EncryptionContext(key, key, key, "coap://[::1]/", coap_ctx)

    with pytest.raises(AccessoryDisconnectedError, match="Request timeout"):
        await asyncio.wait_for(ctx.post(OpCode.CHAR_READ, PAIRINGS_IID, b"", timeout=0.05), 2.0)

    assert coap_ctx.shutdowns == 1
    assert ctx.coap_ctx is None
