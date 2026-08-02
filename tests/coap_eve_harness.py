"""A stand-in for an accessory that does not implement the 0x09 bulk read.

Every bug this transport has produced on real hardware came from a path nobody
thought to exercise, so the point of this harness is that it is driven by the
*whole* public surface rather than by whichever call the author had in mind.
Tests here assert invariants -- properties that must hold on every path -- so a
new entry point inherits the checks for free.

The behaviours modelled are the ones an Eve Room actually exhibits:

* 0x09 is dropped silently: no reply, and the encrypted session dies with it,
  so whatever runs next has to pair-verify again before it can read anything.
* pair-verify is missed when the accessory is asleep, which it frequently is
  immediately after pair-setup -- the moment when giving up is most expensive.
* instance ids are contiguous and modest (2..59, one gap run of 24), so a walk
  terminates on the miss counter rather than the scan limit.

`FakeEve` counts and gates each of those so a test can say "sleep through the
first N pair-verifies" or "answer 0x09 slowly" and then drive any entry point.
"""

from __future__ import annotations

import asyncio
import struct

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import CoAPHomeKitConnection
from aiohomekit.controller.coap.pdu import OpCode, PDUStatus
from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.exceptions import AccessoryDisconnectedError
from aiohomekit.protocol.tlv import HAP_TLV, TLV
from aiohomekit.zeroconf import HomeKitService

ACCESSORY_INFORMATION = 0x3E
PAIRING_SERVICE = 0x55
PAIRINGS_CHARACTERISTIC = 0x50
PAIRINGS_IID = 18
ADVERTISED_NAME = "Eve Room 4B8F"
ACCESSORY_ID = "FA:73:9C:4A:A2:3C"


def signature(char_type: int, svc_type: int, svc_iid: int, fmt: int = 0x04) -> bytes:
    return CharacteristicTLV(
        type=char_type,
        properties=0x10,
        presentation_format=struct.pack("<BxHxxx", fmt, 0x2700),
        service_type=svc_type.to_bytes(16, "little"),
        service_instance_id=svc_iid.to_bytes(2, "little"),
    ).encode()


# Accessory Information low, the Pairing service inside the bounded range, and a
# sensor service above it -- so a bounded read and a full walk see different
# databases, which is what several invariants turn on.
EVE_LAYOUT: dict[int, bytes] = {
    2: signature(0x14, ACCESSORY_INFORMATION, 1),
    3: signature(0x20, ACCESSORY_INFORMATION, 1),
    5: signature(0x23, ACCESSORY_INFORMATION, 1),
    17: signature(0x4C, PAIRING_SERVICE, 16),
    PAIRINGS_IID: signature(PAIRINGS_CHARACTERISTIC, PAIRING_SERVICE, 16),
    40: signature(0x11, 0x96, 39),
    41: signature(0x10, 0x96, 39),
    59: signature(0x21, 0x8A, 58),
}


def value_body(raw: bytes) -> bytes:
    # bytes, not bytearray: real PDU bodies are slices of the decrypted response.
    return bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, raw)]))


class FakeEve:
    """The encryption context an Eve would present.

    `pair_verifies_missed` is how many pair-verify attempts sleep through before
    the accessory answers; `gatt` decides what 0x09 does.
    """

    def __init__(
        self,
        layout: dict[int, bytes] | None = None,
        gatt: str = "dropped",
        gatt_body: bytes | None = None,
        gatt_status: PDUStatus = PDUStatus.UNSUPPORTED_PDU,
        gatt_delay: float = 0.0,
        values: dict[int, bytes] | None = None,
        rtt: float = 0.5,
    ):
        self.layout = EVE_LAYOUT if layout is None else layout
        self.gatt = gatt
        self.gatt_body = gatt_body
        self.gatt_status = gatt_status
        self.gatt_delay = gatt_delay
        self.values = values or {}
        # Starts disconnected, exactly like a real context before pair-verify:
        # a harness that reports is_connected before anyone verified would make
        # every "did this path leave a usable session" check meaningless.
        self.coap_ctx = None

        self.probes: list[float] = []
        self.walked: list[int] = []
        self.sig_timeouts: list[float] = []
        self.writes: list[tuple[int, bytes]] = []
        self.reads: list[int] = []
        # Ordered log of every request, so a test can assert what an operation
        # COST rather than only what it produced. The counts are the point: on
        # this hardware a request is ~0.5 s, and the defect that orphaned a real
        # device was 37 correct requests where 2 were affordable.
        self.requests: list[tuple[OpCode, int]] = []
        # Virtual time. Nothing sleeps -- each request advances a counter -- so
        # "elapsed" is assertable and deterministic under CI load.
        self.rtt = rtt
        self.elapsed = 0.0

    def total_requests(self) -> int:
        return len(self.requests)

    def requests_before_first_write(self) -> int:
        """Cost to reach the commit point.

        For a removal the write of RemovePairing M1 is the moment the accessory
        acts; everything before it is discovery the caller is paying for.
        """
        for index, (opcode, _) in enumerate(self.requests):
            if opcode is OpCode.CHAR_WRITE:
                return index
        return len(self.requests)

    async def post(self, opcode, iid, data, timeout=16.0, expected_statuses=()):
        self.requests.append((opcode, iid))
        self.elapsed += self.rtt
        if opcode is OpCode.UNK_09_READ_GATT:
            self.probes.append(timeout)
            if self.gatt_delay:
                # Slow but capable: honours the caller's timeout.
                if self.gatt_delay >= timeout:
                    self.coap_ctx = None
                    raise AccessoryDisconnectedError("Request timeout")
                await asyncio.sleep(0)
            if self.gatt == "dropped":
                # No reply, and the session goes with it.
                self.coap_ctx = None
                raise AccessoryDisconnectedError("Request timeout")
            if self.gatt == "desync":
                # Transient: a crypto desync says nothing about capability.
                from aiohomekit.exceptions import EncryptionError

                self.coap_ctx = None
                raise EncryptionError("Decryption of PDU POST response failed")
            if self.gatt == "body":
                return (len(self.gatt_body), self.gatt_body)
            return (0, self.gatt_status)

        if opcode is OpCode.CHAR_SIG_READ:
            self.walked.append(iid)
            self.sig_timeouts.append(timeout)
            if iid in self.layout:
                body = self.layout[iid]
                return (len(body), body)
            return (0, PDUStatus.INVALID_INSTANCE_ID)

        if opcode is OpCode.CHAR_WRITE:
            self.writes.append((iid, data))
            if iid not in self.layout:
                # A real accessory rejects a write to an iid it does not have.
                # Answering every write with success made a whole class of
                # defect -- writing to the wrong characteristic -- inexpressible.
                return (0, PDUStatus.INVALID_INSTANCE_ID)
            return (0, b"")

        if opcode is OpCode.CHAR_READ:
            self.reads.append(iid)
            m2 = bytes(
                TLV.encode_list(
                    [(HAP_TLV.kTLVHAPParamValue, bytes(TLV.encode_list([(TLV.kTLVType_State, TLV.M2)])))]
                )
            )
            return (len(m2), m2)

        raise AssertionError(f"unexpected opcode {opcode}")

    def decrypt_event(self, payload: bytes) -> bytes:
        # The event path's crypto is not what these tests are about; the
        # accessory is modelled as always sending a well-formed event.
        return payload

    async def post_all(self, opcode, iids, data):
        if opcode is OpCode.CHAR_READ or opcode is OpCode.UNK_09_READ_GATT:
            return [self.values.get(iid, PDUStatus.INVALID_REQUEST) for iid in iids]
        return [self.values.get(iid, PDUStatus.INVALID_REQUEST) for iid in iids]


def description(config_num: int = 2, state_num: int = 1) -> HomeKitService:
    return HomeKitService(
        name=ADVERTISED_NAME,
        id=ACCESSORY_ID.lower(),
        model="Eve Room 20EBX9901",
        feature_flags=2,
        status_flags=0,
        config_num=config_num,
        state_num=state_num,
        category=10,
        protocol_version="1.2",
        type="_hap._udp.local.",
        address="fdc8::1",
        addresses=["fdc8::1"],
        port=5683,
    )


PAIRING_SERVICE_UUID = "00000055-0000-1000-8000-0026BB765291"
PAIRINGS_CHAR_UUID = "00000050-0000-1000-8000-0026BB765291"


def _pairing_accessory(aid: int, pairings_iid: int, perms: list[str]) -> dict:
    return {
        "aid": aid,
        "services": [
            {
                "iid": 16,
                "type": PAIRING_SERVICE_UUID,
                "characteristics": [
                    {
                        "iid": pairings_iid,
                        "type": PAIRINGS_CHAR_UUID,
                        "perms": perms,
                        "format": "tlv8",
                    }
                ],
            }
        ],
    }


def bridged_cached_map(decoy_iid: int = 250) -> dict:
    """A map whose FIRST accessory is not the primary one.

    HAP addresses the Pairing service on accessory 1, and the wire carries no
    aid -- the write goes to a bare instance id. Reaching for accessories[0]
    instead of aid 1 therefore writes a RemovePairing payload at whatever iid a
    bridged accessory happens to use. Index-vs-aid is the transcription hazard
    here, so the fixture makes the two differ.
    """
    return {
        "config_num": -1,
        "accessories": [
            _pairing_accessory(2, decoy_iid, ["pr", "pw"]),
            _pairing_accessory(1, PAIRINGS_IID, ["pr", "pw"]),
        ],
    }


def read_only_cached_map() -> dict:
    """A cached Pairings characteristic that cannot be written.

    A cache can be wrong about more than the instance id. Writing M1 to a
    characteristic the accessory will not accept a write on wastes the one
    request that matters.
    """
    return {"config_num": -1, "accessories": [_pairing_accessory(1, PAIRINGS_IID, ["pr"])]}


def cached_map(pairings_iid: int = PAIRINGS_IID, config_num: int = -1) -> dict:
    """An entity map of the shape a controller restores before a removal.

    This is the input the harness used to throw away. Both fixtures hard-coded
    `accessories: None` on their fake owner, so every test modelled a *cold*
    pairing -- but a removal is always *warm*, because a controller cannot
    delete a config entry that never had an entity map. Nulling it made the one
    condition under which the production code is correct the only condition
    ever tested.

    Mirrors a real Eve Room: one accessory, the Pairing service carrying a
    writable Pairings characteristic. `config_num=-1` by default because that is
    the marker this repo persists walk-built databases under, so the default
    exercises the least-trusted cache we ever store.
    """
    return {
        "config_num": config_num,
        "accessories": [
            {
                "aid": 1,
                "services": [
                    {
                        "iid": 16,
                        "type": PAIRING_SERVICE_UUID,
                        "characteristics": [
                            {
                                "iid": pairings_iid,
                                "type": PAIRINGS_CHAR_UUID,
                                "perms": ["pr", "pw"],
                                "format": "tlv8",
                            }
                        ],
                    }
                ],
            }
        ],
    }


PAIRING_DATA = {
    "AccessoryPairingID": ACCESSORY_ID,
    "AccessoryIP": "fdc8::1",
    "AccessoryPort": 5683,
    "Connection": "CoAP",
    "iOSPairingId": "some-controller-id",
}


class FakeCharCache:
    def __init__(self):
        self.saved: list[tuple[str, int]] = []
        self.map = None

    def get_map(self, pairing_id):
        return self.map

    def async_create_or_update_map(self, pairing_id, config_num, accessories, *a, **kw):
        self.saved.append((pairing_id, config_num))


class FakeController:
    def __init__(self):
        self._char_cache = FakeCharCache()
        self.pairings = {}
        self.aliases = {}
        self.discoveries = {}


def build_connection(
    eve: FakeEve, sleepy_verifies: int = 0, session: bool = True
) -> CoAPHomeKitConnection:
    """A connection wired to `eve`, whose pair-verify sleeps `sleepy_verifies` times.

    `session=True` starts with a live encrypted session, which is the state
    every caller below connect() assumes. `session=False` starts genuinely
    disconnected, so a test can observe whether an entry point establishes one
    -- is_connected must never be true before a pair-verify has happened.
    """
    owner = type("Owner", (), {"accessories": None, "event_received": lambda *a: None})()
    conn = CoAPHomeKitConnection(owner, "fdc8::1", 5683)
    conn.enc_ctx = eve
    conn._pairing_data = dict(PAIRING_DATA)
    conn.verify_attempts = 0
    if session:
        eve.coap_ctx = object()
    remaining = {"n": sleepy_verifies}

    async def fake_pair_verify(pairing_data):
        conn.verify_attempts += 1
        if remaining["n"] > 0:
            remaining["n"] -= 1
            raise asyncio.TimeoutError
        eve.coap_ctx = object()

    conn.do_pair_verify = fake_pair_verify
    return conn


def build_pairing(
    eve: FakeEve,
    sleepy_verifies: int = 0,
    with_description: bool = True,
    session: bool = False,
    cached_accessories: dict | None = None,
) -> CoAPPairing:
    """A CoAPPairing over `eve`, built the way the controller builds one.

    Defaults to no session: a pairing handed to an entry point should have to
    establish one, so the entry point's own behaviour is what is observed.

    `cached_accessories` is the stored entity map, installed BEFORE construction
    because AbstractPairing.__init__ reads it -- pass `cached_map()` to model a
    warm pairing, which is what a removal always is in the field.
    """
    controller = FakeController()
    if cached_accessories is not None:
        controller._char_cache.map = cached_accessories
    desc = description() if with_description else None
    pairing = CoAPPairing(controller, dict(PAIRING_DATA), description=desc)
    conn = build_connection(eve, sleepy_verifies=sleepy_verifies, session=session)
    conn.owner = pairing
    pairing.connection = conn
    return pairing
