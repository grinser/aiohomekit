#
# Copyright 2022 aiohomekit team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import asyncio
import logging
import random
import struct
import uuid
from collections.abc import Collection, Iterable
from typing import Any

from aiocoap import Context, Message, resource
from aiocoap.error import Error as AiocoapError
from aiocoap.error import NetworkError
from aiocoap.numbers.codes import Code
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from aiohomekit.exceptions import (
    AccessoryDisconnectedError,
    AuthenticationError,
    EncryptionError,
    UnknownError,
)
from aiohomekit.protocol import (
    get_session_keys,
    perform_pair_setup_part1,
    perform_pair_setup_part2,
)
from aiohomekit.protocol.tlv import HAP_TLV, K_TLV_ERROR_NAMES, TLV
from aiohomekit.utils import asyncio_timeout

from ..ble.structs import Characteristic as CharacteristicTLV
from .pdu import (
    OpCode,
    PDUStatus,
    decode_all_pdus,
    decode_pdu,
    encode_all_pdus,
    encode_pdu,
)
from .structs import (
    Pdu09Accessory,
    Pdu09AccessoryContainer,
    Pdu09Characteristic,
    Pdu09CharacteristicContainer,
    Pdu09Database,
    Pdu09Service,
    Pdu09ServiceContainer,
)

logger = logging.getLogger(__name__)

# Short: removing our own pairing ends the session, so this read often gets no
# answer, and the caller should not wait a full request timeout to find out.
REMOVE_PAIRING_M2_TIMEOUT = 4.0

DEFAULT_POST_TIMEOUT = 16.0
# A probe that times out latches 0x09 off for the life of the connection, so this
# must stay at least as generous as post_bytes' default: an accessory that answers
# a large database slowly is supported, not broken.
GATT_PROBE_TIMEOUT = 20.0
# The walk result is cached, so it favours completeness over speed: a wide iid
# range, and the database is only assumed to end after a long run of gaps.
SIGNATURE_WALK_MAX_IID = 300
SIGNATURE_WALK_MAX_MISSES = 25
# Far enough to reach the Pairing service, which is all list_pairings and
# remove_pairing need: HAP lays out an accessory's mandatory services first,
# and on real Eve firmware the Pairings characteristic sits in the first few
# instance ids. Above SIGNATURE_WALK_MAX_MISSES so a bounded read can still
# terminate on the miss counter rather than always running to the bound.
#
# This narrows the unpair window rather than closing it (~76 s for a full walk
# against ~40 s bounded, measured on an Eve Room; a controller may only wait a
# few seconds): the real fix is not enumerating at all for pairing operations.
PAIRING_SERVICE_MAX_IID = 32
# 0x09 returns the entire database, which is the opposite of what a bounded
# read wants, but it is one round-trip when it works. Give it a short window
# rather than the full probe timeout: on firmware that drops it, the wait is
# pure latency on an operation the controller is already timing.
PAIRING_PROBE_TIMEOUT = 4.0
# First accessory's instance id; bridges increment from here.
COAP_ACCESSORY_IID = 1
# Probing a contiguous iid range legitimately misses; not worth a warning.
_WALK_EXPECTED_STATUSES = frozenset(
    {PDUStatus.INVALID_INSTANCE_ID, PDUStatus.INVALID_REQUEST, PDUStatus.UNSUPPORTED_PDU}
)
_WALK_SKIPPABLE_STATUSES = frozenset(
    {PDUStatus.INSUFFICIENT_AUTHENTICATION, PDUStatus.INSUFFICIENT_AUTHORIZATION}
)
# A second one of these marks the start of another accessory.
_ACCESSORY_INFORMATION_SERVICE = 0x3E
# 0x09 reports base-range types in short form and lookups are written against
# that, but signatures always carry the full 128-bit UUID.
_HAP_BASE_UUID = uuid.UUID("00000000-0000-1000-8000-0026BB765291").int
_HAP_BASE_SUFFIX_MASK = (1 << 96) - 1


def _shorten_type(type_: int) -> int:
    """Return the short HomeKit type for a full base UUID, else the value as-is."""
    if type_ & _HAP_BASE_SUFFIX_MASK == _HAP_BASE_UUID & _HAP_BASE_SUFFIX_MASK:
        return type_ >> 96
    return type_


# How a dropped 0x09 surfaces: no reply at all, a 404 whose response then fails
# to decrypt, or the transport being torn down mid-request.
_PROBE_FAILURES: tuple[type[BaseException], ...] = (
    AccessoryDisconnectedError,
    EncryptionError,
    asyncio.TimeoutError,
    AiocoapError,
    # decode_pdu unpacks the header and builds a PDUStatus outside any try, so a
    # short or garbage reply surfaces as one of these rather than a status.
    struct.error,
    ValueError,
)

# Only these mean "this accessory does not implement 0x09". Anything else is
# transient (busy, desynced, unauthenticated) and must not latch it off.
_GATT_UNSUPPORTED_STATUSES = frozenset(
    {PDUStatus.UNSUPPORTED_PDU, PDUStatus.INVALID_REQUEST, PDUStatus.INVALID_INSTANCE_ID}
)


def decode_pdu_03(buf):
    return bytes(dict(TLV.decode_bytes(buf)).get(HAP_TLV.kTLVHAPParamValue))


def decode_list_pairings_response(buf):
    inner_bytes = decode_pdu_03(buf)
    return TLV.decode_bytes(inner_bytes)


class EncryptionContext:
    coap_ctx: Context
    lock: asyncio.Lock
    uri: str

    event_ctr: int
    event_ctx: ChaCha20Poly1305
    recv_ctr: int
    recv_ctx: ChaCha20Poly1305
    send_ctr: int
    send_ctx: ChaCha20Poly1305

    def __init__(self, recv_ctx, send_ctx, event_ctx, uri, coap_ctx):
        self.recv_ctr = 0
        self.recv_ctx = recv_ctx
        self.send_ctr = 0
        self.send_ctx = send_ctx
        self.event_ctr = 0
        self.event_ctx = event_ctx

        self.coap_ctx = coap_ctx
        self.lock = asyncio.Lock()
        self.uri = uri

    def decrypt(self, enc_data: bytes) -> bytes:
        logger.debug("DECRYPT counter=%d" % (self.recv_ctr,))
        dec_data = self.recv_ctx.decrypt(struct.pack("=4xQ", self.recv_ctr), enc_data, b"")
        self.recv_ctr += 1
        return dec_data

    def decrypt_event(self, enc_data: bytes) -> bytes:
        dec_data = self.event_ctx.decrypt(struct.pack("=4xQ", self.event_ctr), enc_data, b"")
        self.event_ctr += 1
        return dec_data

    def encrypt(self, dec_data: bytes) -> bytes:
        logger.debug("ENCRYPT counter=%d" % (self.send_ctr,))
        enc_data = self.send_ctx.encrypt(struct.pack("=4xQ", self.send_ctr), dec_data, b"")
        self.send_ctr += 1
        return enc_data

    async def _decrypt_response(self, response: Message):
        try:
            return self.decrypt(response.payload)
        except InvalidTag:
            logger.error("Decryption failed, desynchronized? Counter=%d/%d" % (self.recv_ctr, self.send_ctr))

            # look back a few counter values
            rewind = min(5, self.recv_ctr)
            self.recv_ctr -= rewind
            for i in range(rewind):
                logger.debug("Attempting to recover by rewind, try %d" % (i + 1,))
                try:
                    return self.decrypt(response.payload)
                except InvalidTag:
                    self.recv_ctr += 1

            # fast forward a few counter values
            for i in range(5):
                logger.debug("Attempting to recover, try %d" % (i + 1,))
                try:
                    # attempt to resynchronize by moving the counter forward
                    # we've got to roll it forward ourselves as the exception prevents that
                    self.recv_ctr += 1
                    return self.decrypt(response.payload)
                except InvalidTag:
                    pass

            # try zeroing out the counters
            try:
                self.recv_ctr = 0
                self.send_ctr = 0
                return self.decrypt(response.payload)
            except InvalidTag:
                pass

            logger.error("Failed flailing attempts to resynchronize, self-destructing in 3, 2, 1...")

            if self.coap_ctx:
                await self.coap_ctx.shutdown()
                self.coap_ctx = None

            raise EncryptionError("Decryption of PDU POST response failed")

    async def post_bytes(self, payload: bytes, timeout: float = DEFAULT_POST_TIMEOUT):
        async with self.lock:
            payload = self.encrypt(payload)

            try:
                request = Message(code=Code.POST, payload=payload, uri=self.uri)
                async with asyncio_timeout(timeout):
                    response = await self.coap_ctx.request(request).response
            except (NetworkError, asyncio.TimeoutError):
                logger.debug("%s: Did not receive a reply; end of session.", self.uri)
                if self.coap_ctx:
                    await self.coap_ctx.shutdown()
                    self.coap_ctx = None
                raise AccessoryDisconnectedError("Request timeout")

            if response.code == Code.NOT_FOUND:
                # maybe the accessory lost power or was otherwise rebooted
                logger.debug("CoAP POST returned 404, our session is gone.")
                await self.coap_ctx.shutdown()
                self.coap_ctx = None
            elif response.code != Code.CHANGED:
                logger.warning(f"CoAP POST returned unexpected code {response}")

            return await self._decrypt_response(response)

    async def post(
        self,
        opcode: OpCode,
        iid: int,
        data: bytes,
        timeout: float = DEFAULT_POST_TIMEOUT,
        expected_statuses: Collection[PDUStatus] = (),
    ) -> tuple[int, bytes | PDUStatus]:
        tid = random.randint(1, 254)
        req_pdu = encode_pdu(opcode, tid, iid, data)
        res_pdu = await self.post_bytes(req_pdu, timeout)
        return decode_pdu(tid, res_pdu, expected_statuses)

    async def post_all(self, opcode: OpCode, iids: list[int], data: list[bytes]) -> list[bytes | PDUStatus]:
        req_pdu = encode_all_pdus(opcode, iids, data)
        res_pdu = await self.post_bytes(req_pdu)
        return decode_all_pdus(0, res_pdu)


class EventResource(resource.Resource):
    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    async def render_put(self, request):
        try:
            payload = self.connection.enc_ctx.decrypt_event(request.payload)
        except InvalidTag:
            logger.debug(
                "Event decryption failed, desynchronized? Counter=%d" % (self.connection.enc_ctx.event_ctr,)
            )
            # XXX invalidate subscriptions, etc
            return Message(code=Code.NOT_FOUND)

        logger.debug(f"CoAP event: {payload.hex()}")

        offset = 0
        while True:
            _, iid, body_len = struct.unpack("<BHH", payload[offset : offset + 5])
            body = payload[offset + 5 : offset + 5 + body_len]

            characteristic = self.connection.info.find_characteristic_by_iid(iid)
            value = decode_pdu_03(body) if body_len > 0 else b""
            if characteristic is not None and body_len > 0:
                characteristic.raw_value = value
                value = characteristic.value
            logger.debug("event ?/%d = %r" % (iid, value))

            if self.connection.owner:
                # XXX aid
                key = (1, iid)
                self.connection.owner.event_received(
                    {
                        key: {
                            "value": value,
                        }
                    }
                )

            offset += 5 + body_len
            if offset >= len(payload):
                break

        return Message(code=Code.VALID)


class CoAPHomeKitConnection:
    def __init__(self, owner, host, port):
        self.address = f"[{host}]:{port}"
        self.connection_lock = asyncio.Lock()
        self.enc_ctx = None
        self.info = None
        self.owner = owner
        self.pair_setup_client = None
        self._pairing_data = None
        # Never cleared: re-probing costs a timeout *and* tears the session down
        # again on affected firmware. Latching on a one-off failure only costs
        # speed, since the walk is plain HAP and works on any accessory.
        self._gatt_unsupported = False
        # Serialises pair-verify so a re-verify cannot race connect() and leave
        # one of two sessions unreferenced (and unclosed) on the accessory.
        self._verify_lock = asyncio.Lock()
        # Serialises enumeration. Two callers can legitimately enumerate at once
        # (config-entry setup and a config-changed notification), and a walk makes
        # that window ~300 round-trips wide instead of one.
        self._enumeration_lock = asyncio.Lock()
        # Set when the walk stopped on the miss counter or ran out of range, i.e.
        # whenever the database was rebuilt rather than read in one request.
        self.database_is_partial = False
        # True when the database came from a signature walk. A walk infers the end
        # of the database from a run of missing instance ids, which is a guess: an
        # accessory numbered sparsely (real CoAP dumps in this repo's fixtures run
        # to iid 64087 with gaps of 43515) looks finished long before it is. Such a
        # database is usable but must never be trusted as authoritative.
        self.database_from_walk = False

    async def reconnect_soon(self):
        if not self.enc_ctx:
            return
        if self.is_connected:
            await self.enc_ctx.coap_ctx.shutdown()
        self.enc_ctx = None
        # _pairing_data is kept: an endpoint change does not change the
        # credentials, and _reverify_session needs them to rebuild the session.
        # XXX can't .connect here w/o pairing_data

    async def do_identify(self):
        client = await Context.create_client_context()
        uri = "coap://%s/0" % (self.address)

        request = Message(code=Code.POST, payload=b"", uri=uri)
        async with asyncio_timeout(4.0):
            response = await client.request(request).response

        await client.shutdown()
        client = None

        return response.code == Code.CHANGED

    async def do_pair_setup(self, with_auth):
        self.pair_setup_client = await Context.create_client_context()
        uri = "coap://%s/1" % (self.address)
        logger.debug(f"Pair setup 1/2 uri={uri}")

        state_machine = perform_pair_setup_part1(with_auth)
        request, expected = state_machine.send(None)
        while True:
            try:
                payload = TLV.encode_list(request)
                request = Message(code=Code.POST, payload=payload, uri=uri)
                # some operations can take some time
                async with asyncio_timeout(16.0):
                    response = await self.pair_setup_client.request(request).response
                payload = TLV.decode_bytes(response.payload, expected=expected)

                request, expected = state_machine.send(payload)
            except StopIteration as result:
                salt, srpB = result.value
                return salt, srpB
            except Exception:
                logger.debug("Pair setup 1/2 failed!")
                await self.pair_setup_client.shutdown()
                raise

    async def do_pair_setup_finish(self, pin, salt, srpB):
        uri = "coap://%s/1" % (self.address)
        logger.debug(f"Pair setup 2/2 uri={uri}")

        state_machine = perform_pair_setup_part2(pin, str(uuid.uuid4()), salt, srpB)
        request, expected = state_machine.send(None)
        while True:
            try:
                payload = TLV.encode_list(request)
                request = Message(code=Code.POST, payload=payload, uri=uri)
                async with asyncio_timeout(16.0):
                    response = await self.pair_setup_client.request(request).response

                payload = TLV.decode_bytes(response.payload, expected=expected)

                request, expected = state_machine.send(payload)
            except StopIteration as result:
                pairing = result.value
                break
            except Exception:
                logger.debug("Pair setup 2/2 failed!")
                await self.pair_setup_client.shutdown()
                raise

        logger.debug(f"Paired with CoAP HAP accessory at {self.address}!")
        await self.pair_setup_client.shutdown()
        self.pair_setup_client = None

        return pairing

    async def do_pair_verify(self, pairing_data):
        # Remembered so _read_gatt_database can pair-verify again: a dropped 0x09
        # kills the session, and the walk that follows needs a live one.
        self._pairing_data = pairing_data
        if self.is_connected:
            logger.debug("Connecting to connected device?")
            await self.enc_ctx.coap_ctx.shutdown()
            self.enc_ctx = None

        root = resource.Site()
        coap_client = await Context.create_server_context(root, bind=("::", 0))
        uri = "coap://%s/2" % (self.address)
        logger.debug(f"Pair verify uri={uri}")

        state_machine = get_session_keys(pairing_data)

        request, expected = state_machine.send(None)
        while True:
            try:
                payload = TLV.encode_list(request)
                request = Message(code=Code.POST, payload=payload, uri=uri)
                async with asyncio_timeout(8.0):
                    response = await coap_client.request(request).response

                payload = TLV.decode_bytes(response.payload, expected=expected)

                request, expected = state_machine.send(payload)
            except StopIteration as result:
                _, derive = result.value
                break
            except Exception:
                # clean up coap context
                await coap_client.shutdown()
                coap_client = None
                # re-raise any exception
                raise

        recv_key = derive(b"Control-Salt", b"Control-Read-Encryption-Key")
        recv_ctx = ChaCha20Poly1305(recv_key)
        send_key = derive(b"Control-Salt", b"Control-Write-Encryption-Key")
        send_ctx = ChaCha20Poly1305(send_key)
        event_key = derive(b"Event-Salt", b"Event-Read-Encryption-Key")
        event_ctx = ChaCha20Poly1305(event_key)

        uri = "coap://%s/" % (self.address)

        self.enc_ctx = EncryptionContext(recv_ctx, send_ctx, event_ctx, uri, coap_client)

        logger.debug(f"Connected to CoAP HAP accessory at {self.address}!")
        root.add_resource([], EventResource(self))

        return True

    async def connect(self, pairing_data, enumerate_database: bool = True):
        async with self.connection_lock:
            if self.is_connected:
                logger.debug("Already connected")
                return

            try:
                async with self._verify_lock:
                    # Re-check under the lock: another task may have established
                    # the session while we waited, and verifying again would shut
                    # its context down and replace it.
                    if not self.is_connected:
                        await self.do_pair_verify(pairing_data)
            except asyncio.TimeoutError:
                logger.debug("Pair verify timed out")
                raise AccessoryDisconnectedError("Pair verify timed out")
            except Exception as exc:
                logger.debug("Pair verify failed", exc_info=exc)
                raise AccessoryDisconnectedError("Pair verify failed")

            if enumerate_database:
                # Needed to read/write characteristics -- but a pairing
                # operation needs exactly one characteristic and finds it
                # itself, so making it wait out a full enumeration here is what
                # pushes an unpair past the controller's patience. Measured on
                # an Eve Room: a cold remove_pairing spent 20 s on the 0x09
                # probe and 23 s in the walk this call started, and was
                # cancelled 0.3 s before the walk would have finished.
                await self.get_accessory_info()

            return

    @property
    def is_connected(self):
        return self.enc_ctx is not None and self.enc_ctx.coap_ctx is not None

    async def _reverify_session(self) -> None:
        """Re-establish a session a dropped 0x09 tore down.

        Serialised against connect(): both run pair-verify and both assign
        enc_ctx, so without this one of the two sessions is left unreferenced,
        leaking a socket and a session slot on an accessory that has few.
        """
        async with self._verify_lock:
            if self.is_connected:
                return
            if self._pairing_data is None:
                raise AccessoryDisconnectedError("Cannot re-establish session: pairing data unavailable")
            await self.do_pair_verify(self._pairing_data)

    async def _signature_walk(self, max_iid: int = SIGNATURE_WALK_MAX_IID) -> tuple[dict[int, bytes], bool]:
        """Enumerate the accessory database by reading each characteristic's
        signature (0x01), used when 0x09 is unavailable. Invalid iids come back
        fast as a PDUStatus, so probing a contiguous range is cheap; stop after a
        long run of gaps.
        """
        signatures: dict[int, bytes] = {}
        misses = 0
        complete = False
        for iid in range(1, max_iid + 1):
            if not self.is_connected:
                raise AccessoryDisconnectedError(f"Session ended during the signature walk at iid {iid}")
            result = await self.enc_ctx.post(
                OpCode.CHAR_SIG_READ,
                iid,
                b"",
                expected_statuses=_WALK_EXPECTED_STATUSES,
            )
            body = result[1] if result is not None else None
            if isinstance(body, (bytes, bytearray)):
                signatures[iid] = bytes(body)
                misses = 0
                continue
            if body in _WALK_SKIPPABLE_STATUSES:
                # A property of this characteristic, not of the session: skip it
                # rather than discarding every signature collected so far.
                misses += 1
                continue
            if body not in _WALK_EXPECTED_STATUSES:
                # Not a gap -- the accessory is busy or the session is desynced.
                # Counting it would end the walk mid-database and cache the result
                # as if it were the whole accessory.
                raise AccessoryDisconnectedError(f"Signature walk failed at iid {iid} with {body!r}")
            misses += 1
            if misses >= SIGNATURE_WALK_MAX_MISSES:
                logger.debug(
                    "Signature walk stopping at iid %d after %d consecutive misses; "
                    "assuming end of database (%d characteristics, last at iid %d)",
                    iid,
                    misses,
                    len(signatures),
                    max(signatures) if signatures else 0,
                )
                complete = True
                break
        if not complete and max_iid == SIGNATURE_WALK_MAX_IID:
            # Ran out of range with no long gap: the accessory may have
            # characteristics above the scan limit that we are about to drop.
            logger.warning(
                "Signature walk reached the iid scan limit (%d); the accessory database may be incomplete",
                SIGNATURE_WALK_MAX_IID,
            )
        return signatures, complete

    def _database_from_signatures(self, signatures: dict[int, bytes]) -> Pdu09Database:
        """Rebuild a Pdu09Database from per-characteristic signature reads.

        Each signature carries its parent service (type + instance id) but not an
        accessory id. Every HAP accessory begins with one Accessory Information
        service, so characteristics are grouped into services and a new accessory
        is started whenever a second Accessory Information service appears (in iid
        order). Single-accessory devices -- the common HAP-over-Thread case --
        produce one accessory (id 1); bridges produce one accessory per
        Accessory Information service.
        """
        accessories: list[tuple[dict[int, Pdu09Service], list[int]]] = []
        services: dict[int, Pdu09Service] = {}
        order: list[int] = []
        # A new accessory is recognised by an Accessory Information service that
        # either carries a different instance id or repeats a characteristic type
        # (bridges may number services per accessory, so the iid alone is not
        # always enough).
        current_info_iid: int | None = None
        current_info_types: set[int] = set()
        decode_failures = 0

        for iid in sorted(signatures):
            try:
                sig = CharacteristicTLV.decode(signatures[iid])
            except Exception as exc:
                decode_failures += 1
                logger.debug("Skipping iid %d, signature decode failed: %r", iid, exc)
                continue

            if not sig.service_type or not sig.service_instance_id:
                # Without a parent service the characteristic cannot be placed;
                # defaulting to service 0 would collapse every such signature
                # into one synthetic service.
                decode_failures += 1
                logger.debug("Skipping iid %d, signature carries no service", iid)
                continue

            svc_type = _shorten_type(int.from_bytes(sig.service_type, "little"))
            svc_iid = int.from_bytes(sig.service_instance_id, "little")
            is_accessory_info = svc_type == _ACCESSORY_INFORMATION_SERVICE

            if is_accessory_info and current_info_iid is not None:
                if svc_iid != current_info_iid or sig.type in current_info_types:
                    accessories.append((services, order))
                    services, order = {}, []
                    current_info_iid, current_info_types = None, set()

            service = services.get(svc_iid)
            if service is None:
                service = Pdu09Service(
                    type=svc_type,
                    instance_id=svc_iid,
                    _characteristics=[],
                    properties=0,
                    linked_services=None,
                )
                services[svc_iid] = service
                order.append(svc_iid)
            service._characteristics.append(
                Pdu09CharacteristicContainer(
                    characteristic=Pdu09Characteristic(
                        type=_shorten_type(sig.type),
                        instance_id=iid,
                        properties=sig.properties,
                        presentation_format=sig.presentation_format,
                        valid_range=sig.valid_range,
                        step_value=sig.step_value,
                        valid_values=sig.valid_values,
                        valid_values_range=sig.valid_values_range,
                        user_descriptor=sig.user_description,
                    )
                )
            )
            if is_accessory_info:
                current_info_iid = svc_iid
                current_info_types.add(sig.type)

        accessories.append((services, order))
        if decode_failures:
            logger.warning(
                "Discarded %d of %d characteristic signatures that failed to decode",
                decode_failures,
                len(signatures),
            )

        containers = []
        aid = COAP_ACCESSORY_IID
        for accessory_services, service_order in accessories:
            if not accessory_services:
                continue
            containers.append(
                Pdu09AccessoryContainer(
                    accessory=Pdu09Accessory(
                        instance_id=aid,
                        _services=[
                            Pdu09ServiceContainer(service=accessory_services[i]) for i in service_order
                        ],
                    )
                )
            )
            aid += 1
        return Pdu09Database(_accessories=containers)

    async def _read_gatt_database(self, max_iid: int = SIGNATURE_WALK_MAX_IID) -> Pdu09Database:
        """Read the accessory database. Prefer the 0x09 bulk read; if the
        accessory does not implement it, rebuild from signature reads.

        Some Thread accessories (e.g. Eve Room, HA #167379) silently drop 0x09
        *and* tear down the secured session, so on a 0x09 timeout we must
        re-establish the session (pair-verify) before reading another way. Once
        an accessory is known not to answer 0x09 the probe is skipped entirely on
        later reads, avoiding both its timeout and the teardown it causes.
        """
        session_alive = True
        body = None
        if not self._gatt_unsupported:
            try:
                probe_timeout = (
                    GATT_PROBE_TIMEOUT if max_iid == SIGNATURE_WALK_MAX_IID else PAIRING_PROBE_TIMEOUT
                )
                _, body = await self.enc_ctx.post(
                    OpCode.UNK_09_READ_GATT, 0x0000, b"", timeout=probe_timeout
                )
            except _PROBE_FAILURES:
                # No reply at all is the signature of firmware that drops 0x09.
                logger.debug("0x09 not answered; will reconnect and rebuild without it")
                session_alive = False
                self._gatt_unsupported = True

            if isinstance(body, (bytes, bytearray)):
                try:
                    info = Pdu09Database.decode(body)
                    logger.debug(f"Get accessory info: {info.to_dict()!r}")
                    # 0x09 returns the whole database in one request, so unlike a
                    # walk this is known complete.
                    self.database_is_partial = False
                    self.database_from_walk = False
                    return info
                except Exception as exc:
                    # Fall back this time, but 0x09 did respond, so it is not
                    # remembered as unsupported.
                    logger.error(f"TLV decode failed: {body.hex()}", exc_info=exc)
            elif body is not None:
                # A status: the session is healthy. Only a definitive rejection
                # means the accessory lacks 0x09 -- busy or desynced is transient,
                # and latching on it would downgrade a capable accessory to a
                # 300-request walk for the life of the connection.
                logger.debug("0x09 returned status %r; rebuilding without it", body)
                if body in _GATT_UNSUPPORTED_STATUSES:
                    self._gatt_unsupported = True

        if not session_alive:
            # The walk and the value reads that follow need a live session.
            await self._reverify_session()

        logger.debug("Rebuilding accessory database via signature reads")
        signatures, walk_complete = await self._signature_walk(max_iid)
        if not signatures:
            raise AccessoryDisconnectedError("Unable to parse accessory database")
        logger.debug("Signature walk found %d characteristics: %s", len(signatures), sorted(signatures))
        info = self._database_from_signatures(signatures)
        if not info.accessories:
            # An empty database would be cached by the controller as a valid but
            # characteristic-less accessory; fail instead and let the caller retry.
            raise AccessoryDisconnectedError(
                "Unable to parse accessory database: no characteristic signature "
                f"of {len(signatures)} could be decoded"
            )

        # A full walk that ends on the scan limit is truncated; so is *any*
        # bounded read, even one that ended on the miss counter -- the bound
        # exists to find one characteristic quickly, and whatever it built must
        # never be published, cached, or reused as the whole accessory.
        self.database_is_partial = (not walk_complete) or (max_iid != SIGNATURE_WALK_MAX_IID)
        self.database_from_walk = True
        return info

    def invalidate_database(self) -> None:
        """Force the next enumeration to re-read rather than reuse."""
        self.info = None

    async def get_accessory_info(self, max_iid: int = SIGNATURE_WALK_MAX_IID):
        """Read the accessory database and every readable value.

        `max_iid` bounds a signature walk, should one be needed. It is used only
        by pairing operations, which need a single characteristic and cannot wait
        out a full walk; the resulting database is always marked partial.
        """
        async with self._enumeration_lock:
            if not self.is_connected:
                # The wait can be long enough for the session to have gone away.
                raise AccessoryDisconnectedError("Connection lost before enumerating")
            return await self._enumerate(max_iid)

    async def _enumerate(self, max_iid: int = SIGNATURE_WALK_MAX_IID):
        if self.info is not None and self.database_from_walk and not self.database_is_partial:
            # Reuse only what a walk built: instance ids do not change without a
            # config-number change (which invalidates this via
            # _process_config_changed), and re-walking on every reconnect would
            # cost ~300 sequential requests on an accessory that drops 0x09 --
            # more than the poll interval, on battery power. An accessory that
            # answers 0x09 keeps re-reading its database on every enumeration,
            # exactly as before this fallback existed.
            logger.debug("Reusing the walk-built accessory database; reading values only")
        else:
            self.info = await self._read_gatt_database(max_iid)

        # read all values
        for accessory in self.info.accessories:
            # one service at a time
            for service in accessory.services:
                # first, collect all readable characteristics
                readable = [char for char in service.characteristics if char.supports_secure_reads]

                # get instance IDs
                iids = [char.instance_id for char in readable]

                # make a list of zero length byte strings
                data = [b""] * len(iids)

                # send the read requests
                results = await self.enc_ctx.post_all(OpCode.CHAR_READ, iids, data)

                for idx, result in enumerate(results):
                    if isinstance(result, bytes):
                        # success, let's convert the value
                        value = decode_pdu_03(result) if len(result) > 0 else b""
                        readable[idx].raw_value = value
                        logger.debug(
                            "Read value for %X.%X iid %d: value %r"
                            % (
                                service.type,
                                readable[idx].type,
                                readable[idx].instance_id,
                                readable[idx].value,
                            )
                        )
                    else:
                        # characteristic wasn't readable
                        logger.debug(
                            "Failed to read %X.%X iid %d"
                            % (
                                service.type,
                                readable[idx].type,
                                readable[idx].instance_id,
                            )
                        )

        return self.info.to_dict()

    def _read_characteristics_exit(
        self, ids: list[tuple[int, int]], pdu_results: list[bytes | PDUStatus]
    ) -> dict:
        results = {}
        for idx, result in enumerate(pdu_results):
            aid_iid = ids[idx]
            if isinstance(result, PDUStatus):
                logger.debug("Failed to read aid %d iid %d" % (int(aid_iid[0]), int(aid_iid[1])))
                results[aid_iid] = {
                    "description": result.description,
                    "status": -result.value,  # XXX
                }
            else:
                # decode TLV to get byte value
                value = decode_pdu_03(result) if len(result) > 0 else b""
                # find characteristic so we can get the data type
                characteristic = self.info.find_characteristic_by_iid(int(aid_iid[1]))
                # if we found it & have a non-empty value...
                if characteristic is not None and len(result) > 0:
                    # set the raw bytes
                    characteristic.raw_value = value
                    # and get the decoded value
                    value = characteristic.value
                # add result to dict
                results[aid_iid] = {
                    "value": value,
                }
                logger.debug(
                    "Read value for aid %d iid %d: value %r"
                    % (
                        int(aid_iid[0]),
                        int(aid_iid[1]),
                        value,
                    )
                )

        logger.debug(f"Read characteristics: {results!r}")
        return results

    async def read_characteristics(
        self, characteristics: Iterable[tuple[int, int]]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Read characteristics from the accessory."""
        # _read_characteristics_exit expects a list of tuples
        # as it does an ordered read so we need to convert
        # to a list to preserve the order
        ids = list(characteristics)
        iids = [int(aid_iid[1]) for aid_iid in characteristics]
        data = [b""] * len(iids)
        pdu_results = await self.enc_ctx.post_all(OpCode.CHAR_READ, iids, data)
        return self._read_characteristics_exit(ids, pdu_results)

    def _write_characteristics_enter(self, ids_values: list[tuple[int, int, Any]]) -> list[bytearray]:
        # convert provided values to appropriate binary format for each characteristic
        tlv_values = []
        for _, aid_iid_value in enumerate(ids_values):
            # look up characteristic
            characteristic = self.info.find_characteristic_by_aid_iid(
                int(aid_iid_value[0]), int(aid_iid_value[1])
            )
            # write value to cache + convert to appropriate binary representation
            characteristic.value = aid_iid_value[2]
            # get the converted value
            value = characteristic.raw_value
            # encode into TLV
            value_tlv = TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, value)])
            # add to list
            tlv_values.append(value_tlv)

        return tlv_values

    def _write_characteristics_exit(
        self,
        ids_values: list[tuple[int, int, Any]],
        pdu_results: list[bytes | PDUStatus],
    ) -> dict:
        # transform results
        # only error conditions are returned
        results = {}
        for idx, result in enumerate(pdu_results):
            aid_iid_value = ids_values[idx]
            key = (aid_iid_value[0], aid_iid_value[1])
            if isinstance(result, PDUStatus):
                results[key] = {
                    "descripton": result.description,
                    "status": -result.value,  # XXX
                }
            else:
                logger.debug(
                    "Wrote value for aid %d iid %d"
                    % (
                        int(aid_iid_value[0]),
                        int(aid_iid_value[1]),
                    )
                )

        return results

    async def write_characteristics(self, ids_values: list[tuple[int, int, Any]]):
        tlv_values = self._write_characteristics_enter(ids_values)

        # batch write
        pdu_results = await self.enc_ctx.post_all(
            OpCode.CHAR_WRITE,
            [int(aid_iid_value[1]) for aid_iid_value in ids_values],
            tlv_values,
        )

        return self._write_characteristics_exit(ids_values, pdu_results)

    def _subscribe_to_exit(self, ids: list[tuple[int, int]], pdu_results: list[bytes | PDUStatus]) -> dict:
        results = {}
        for idx, result in enumerate(pdu_results):
            aid_iid = ids[idx]
            key = (aid_iid[0], aid_iid[1])
            if isinstance(result, PDUStatus):
                results[key] = {
                    "descripton": result.description,
                    "status": -result.value,  # XXX
                }
            else:
                logger.debug(
                    "Subscribed to aid %d iid %d"
                    % (
                        int(aid_iid[0]),
                        int(aid_iid[1]),
                    )
                )

        return results

    async def subscribe_to(self, ids: list[tuple[int, int]]):
        iids = [int(aid_iid[1]) for aid_iid in ids]
        data = [b""] * len(iids)
        pdu_results = await self.enc_ctx.post_all(OpCode.UNK_0B_SUBSCRIBE, iids, data)
        return self._subscribe_to_exit(ids, pdu_results)

    def _unsubscribe_from_exit(
        self, ids: list[tuple[int, int]], pdu_results: list[bytes | PDUStatus]
    ) -> dict:
        results = {}
        for idx, result in enumerate(pdu_results):
            aid_iid = ids[idx]
            key = (aid_iid[0], aid_iid[1])
            if isinstance(result, PDUStatus):
                results[key] = {
                    "descripton": result.description,
                    "status": -result.value,  # XXX
                }
            else:
                logger.debug(
                    "Unsubscribed from aid %d iid %d"
                    % (
                        int(aid_iid[0]),
                        int(aid_iid[1]),
                    )
                )

        return results

    async def unsubscribe_from(self, ids: list[tuple[int, int]]):
        if not ids:
            return {}
        iids = [int(aid_iid[1]) for aid_iid in ids]
        data = [b""] * len(iids)
        pdu_results = await self.enc_ctx.post_all(OpCode.UNK_0C_UNSUBSCRIBE, iids, data)
        return self._unsubscribe_from_exit(ids, pdu_results)

    async def _pairings_characteristic(self):
        """Locate the Pairing service's Pairings characteristic.

        Prefers whatever has already been enumerated, then a bounded read, and
        only then a full one. A full signature walk takes longer than a
        controller will wait to remove a pairing, and giving up leaves the
        pairing orphaned on the accessory. Routed through get_accessory_info so
        the enumeration lock is held: enumerating here directly could clobber a
        full database another caller was still publishing.
        """
        for bound in (None, PAIRING_SERVICE_MAX_IID, SIGNATURE_WALK_MAX_IID):
            if bound is not None:
                await self.get_accessory_info(bound)
            if self.info is not None:
                char = self.info.accessories[0].find_service_characteristic_by_type(0x55, 0x50)
                if char is not None:
                    return char
        raise UnknownError("Accessory exposes no Pairing service")

    async def list_pairings(self):
        pairings_characteristic = await self._pairings_characteristic()

        # list pairings M1
        m1_payload = TLV.encode_list(
            [
                (TLV.kTLVType_State, TLV.M1),
                (TLV.kTLVType_Method, TLV.ListPairings),
            ]
        )
        payload = TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, m1_payload)])
        payload_len, payload = await self.enc_ctx.post(
            OpCode.CHAR_WRITE,
            pairings_characteristic.instance_id,
            payload,
        )
        # XXX check response

        payload_len, payload = await self.enc_ctx.post(
            OpCode.CHAR_READ,
            pairings_characteristic.instance_id,
            b"",
        )
        # XXX check response

        # list pairings M2
        m2 = decode_list_pairings_response(payload)

        m2_state = list(filter(lambda x: x[0] == TLV.kTLVType_State, m2))
        if len(m2_state) != 1 or m2_state[0][1] != TLV.M2:
            logger.debug("Unexpected state in list pairings M2")
            return None

        m2_error = list(filter(lambda x: x[0] == TLV.kTLVType_Error, m2))
        if len(m2_error) != 0:
            logger.debug(f"Error from accessory during list pairings: {m2_error[0][1]}")
            return None

        id_list = [pairing_tuple[1] for pairing_tuple in m2 if pairing_tuple[0] == TLV.kTLVType_Identifier]
        pk_list = [pairing_tuple[1] for pairing_tuple in m2 if pairing_tuple[0] == TLV.kTLVType_PublicKey]
        pr_list = [
            int.from_bytes(pairing_tuple[1], byteorder="little")
            for pairing_tuple in m2
            if pairing_tuple[0] == TLV.kTLVType_Permissions
        ]
        return list(zip(id_list, pk_list, pr_list))

    async def remove_pairing(self, pairing_id) -> bool:
        pairings_characteristic = await self._pairings_characteristic()

        # remove pairings M1
        m1_payload = TLV.encode_list(
            [
                (TLV.kTLVType_State, TLV.M1),
                (TLV.kTLVType_Method, TLV.RemovePairing),
                (TLV.kTLVType_Identifier, pairing_id.encode()),
            ]
        )
        # As with list_pairings() above, kTLVHAPParamParamReturnResponse is not
        # set: it is a BLE-transport write parameter, and this procedure
        # completes without it against real hardware.
        payload = TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, m1_payload)])
        result_len, result = await self.enc_ctx.post(
            OpCode.CHAR_WRITE,
            pairings_characteristic.instance_id,
            payload,
        )

        if isinstance(result, PDUStatus):
            if result in [
                PDUStatus.INSUFFICIENT_AUTHENTICATION,
                PDUStatus.INSUFFICIENT_AUTHORIZATION,
            ]:
                raise AuthenticationError("Remove pairing failed")
            raise UnknownError("Remove pairing failed")

        # The procedure is not complete until M2 is read back: accessories exist
        # (e.g. Eve firmware over Thread) that do not apply the removal until
        # the response is collected, so writing alone leaves the pairing in
        # place while telling the caller it succeeded.
        #
        # An unreadable M2 is not a failure: removing our own pairing ends the
        # session, so the response legitimately may never arrive. M1 was
        # accepted, so the removal stands -- raising here would make the caller
        # keep a local record for a pairing the accessory has already dropped,
        # which is why the broad except below is deliberate.
        try:
            async with asyncio_timeout(REMOVE_PAIRING_M2_TIMEOUT):
                _, result = await self.enc_ctx.post(
                    OpCode.CHAR_READ,
                    pairings_characteristic.instance_id,
                    b"",
                )
        except Exception as exc:
            logger.debug("Remove pairing M2 not read (%r); the removal itself was accepted", exc)
            return True

        if isinstance(result, PDUStatus) or not result:
            logger.debug("Remove pairing M2 not readable (%s); the removal itself was accepted", result)
            return True

        try:
            m2 = decode_list_pairings_response(result)
        except Exception as exc:
            logger.debug("Remove pairing M2 undecodable (%r); the removal itself was accepted", exc)
            return True

        # The accessory did answer: an explicit error in M2 is a real failure.
        m2_error = [entry for entry in m2 if entry[0] == TLV.kTLVType_Error]
        if m2_error:
            code = m2_error[0][1]
            if code == TLV.kTLVError_Authentication:
                raise AuthenticationError("Remove pairing failed")
            raise UnknownError(f"Remove pairing failed: {K_TLV_ERROR_NAMES.get(code[0], 'Unknown')}")

        m2_state = [entry for entry in m2 if entry[0] == TLV.kTLVType_State]
        if len(m2_state) != 1 or m2_state[0][1] != TLV.M2:
            logger.debug("Unexpected state in remove pairing M2: %r", m2_state)

        return True
