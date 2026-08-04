"""Pairing operations resolve their one instance id without enumerating."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aiohomekit.controller.coap.connection import CoAPHomeKitConnection
from aiohomekit.controller.coap.pdu import OpCode
from aiohomekit.controller.coap.structs import (
    Pdu09Accessory,
    Pdu09AccessoryContainer,
    Pdu09Database,
)
from aiohomekit.exceptions import UnknownError
from aiohomekit.model import Accessories, Accessory
from aiohomekit.model.characteristics import CharacteristicsTypes
from aiohomekit.model.services import ServicesTypes
from aiohomekit.protocol.tlv import HAP_TLV, TLV

from .test_controller_coap import database_nanoleaf_bulb

PAIRINGS_IID_IN_FIXTURE = 37


def _accessories(aid: int = 1) -> Accessories:
    """A model database of the shape a controller restores from its cache."""
    accessory = Accessory.create_with_info(
        aid,
        name="Test",
        manufacturer="Test",
        model="Test",
        serial_number="0001",
        firmware_revision="1.0",
    )
    pairing = accessory.add_service(ServicesTypes.PAIRING)
    pairing.add_char(CharacteristicsTypes.PAIRING_PAIRINGS)

    accessories = Accessories()
    accessories.add_accessory(accessory)
    return accessories


class FakeOwner:
    """Stands in for CoAPPairing, which is what owns the connection."""

    def __init__(self, accessories: Accessories | None = None) -> None:
        self.accessories = accessories


def _m2_reply() -> bytes:
    """A well-formed M2 reply in a HAP value TLV; decodes as an empty pairing list and as a confirmed removal."""
    return bytes(
        TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, bytes(TLV.encode_list([(TLV.kTLVType_State, TLV.M2)])))])
    )


def _connection(owner: FakeOwner | None = None) -> CoAPHomeKitConnection:
    connection = CoAPHomeKitConnection(owner, "any", 1234)
    connection.enc_ctx = AsyncMock()
    # These tests are about how the iid is resolved, not the reply protocol.
    m2 = _m2_reply()
    connection.enc_ctx.post.return_value = (len(m2), m2)
    return connection


def test_cached_iid_comes_from_the_owners_accessories() -> None:
    accessories = _accessories()
    expected = accessories.aid(1).services.first(service_type=ServicesTypes.PAIRING)[
        CharacteristicsTypes.PAIRING_PAIRINGS
    ]

    connection = _connection(FakeOwner(accessories))

    assert connection._cached_pairings_iid() == expected.iid


@pytest.mark.parametrize(
    "owner",
    [None, FakeOwner(None), FakeOwner(Accessories())],
    ids=["no owner", "no accessories", "empty accessories"],
)
def test_cached_iid_is_absent_rather_than_raising(owner: FakeOwner | None) -> None:
    """Every caller treats None as "ask the accessory", so this must not raise."""
    assert _connection(owner)._cached_pairings_iid() is None


def test_cached_iid_is_keyed_on_aid_not_index() -> None:
    """A bridge's first accessory is not necessarily aid 1; reading index 0 would address the wrong one."""
    accessories = _accessories(aid=7)

    assert _connection(FakeOwner(accessories))._cached_pairings_iid() is None


def test_cached_iid_ignores_a_characteristic_that_cannot_be_written() -> None:
    """RemovePairing is a write, so a cache entry without write permission is not usable."""
    accessories = _accessories()
    char = accessories.aid(1).services.first(service_type=ServicesTypes.PAIRING)[
        CharacteristicsTypes.PAIRING_PAIRINGS
    ]
    char.perms = ["pr"]

    assert _connection(FakeOwner(accessories))._cached_pairings_iid() is None


async def test_remove_pairing_does_not_enumerate_when_the_cache_has_the_iid() -> None:
    accessories = _accessories()
    connection = _connection(FakeOwner(accessories))
    expected_iid = (
        accessories.aid(1)
        .services.first(service_type=ServicesTypes.PAIRING)[CharacteristicsTypes.PAIRING_PAIRINGS]
        .iid
    )

    with patch.object(connection, "get_accessory_info", AsyncMock()) as get_accessory_info:
        assert await connection.remove_pairing("some-pairing-id") is True

    get_accessory_info.assert_not_awaited()
    assert connection.info is None

    opcodes = [call.args[0] for call in connection.enc_ctx.post.await_args_list]
    assert OpCode.UNK_09_READ_GATT not in opcodes
    assert connection.enc_ctx.post.await_args_list[0].args[:2] == (
        OpCode.CHAR_WRITE,
        expected_iid,
    )


async def test_list_pairings_does_not_enumerate_when_the_cache_has_the_iid() -> None:
    accessories = _accessories()
    connection = _connection(FakeOwner(accessories))
    expected_iid = (
        accessories.aid(1)
        .services.first(service_type=ServicesTypes.PAIRING)[CharacteristicsTypes.PAIRING_PAIRINGS]
        .iid
    )

    with patch.object(connection, "get_accessory_info", AsyncMock()) as get_accessory_info:
        result = await connection.list_pairings()

    assert result == []
    get_accessory_info.assert_not_awaited()
    assert connection.info is None

    opcodes = [call.args[0] for call in connection.enc_ctx.post.await_args_list]
    assert OpCode.UNK_09_READ_GATT not in opcodes
    assert connection.enc_ctx.post.await_args_list[0].args[:2] == (
        OpCode.CHAR_WRITE,
        expected_iid,
    )


async def test_pairing_operations_fall_back_to_enumerating() -> None:
    """A pairing built by hand has no cache behind it. It still has to work."""
    connection = _connection(FakeOwner(None))

    async def _enumerate() -> None:
        connection.info = Pdu09Database.decode(database_nanoleaf_bulb)

    with patch.object(connection, "get_accessory_info", AsyncMock(side_effect=_enumerate)):
        assert await connection._pairings_iid() == PAIRINGS_IID_IN_FIXTURE


async def test_enumerating_once_is_enough() -> None:
    connection = _connection(FakeOwner(None))
    connection.info = Pdu09Database.decode(database_nanoleaf_bulb)

    with patch.object(connection, "get_accessory_info", AsyncMock()) as get_accessory_info:
        assert await connection._pairings_iid() == PAIRINGS_IID_IN_FIXTURE

    get_accessory_info.assert_not_awaited()


async def test_a_database_without_an_accessory_at_aid_1_is_reported() -> None:
    connection = _connection(FakeOwner(None))
    connection.info = Pdu09Database(_accessories=[])

    with pytest.raises(UnknownError):
        await connection._pairings_iid()


async def test_an_accessory_without_a_pairings_characteristic_is_reported() -> None:
    connection = _connection(FakeOwner(None))
    connection.info = Pdu09Database(
        _accessories=[Pdu09AccessoryContainer(accessory=Pdu09Accessory(instance_id=1, _services=[]))]
    )

    with pytest.raises(UnknownError):
        await connection._pairings_iid()


@pytest.mark.parametrize(
    ("method", "args"),
    # aid 1 / iid 51 is the On characteristic of the bulb in the fixture.
    [("read_characteristics", ([(1, 51)],)), ("write_characteristics", ([(1, 51, True)],))],
)
async def test_an_ordinary_read_or_write_enumerates_a_session_that_did_not(method: str, args: tuple) -> None:
    """A read or write on a session a pairing operation opened without a database enumerates first."""
    connection = _connection(FakeOwner(_accessories()))
    connection.enc_ctx.post_all.return_value = []
    assert connection.info is None

    async def _enumerate() -> None:
        connection.info = Pdu09Database.decode(database_nanoleaf_bulb)

    with patch.object(connection, "get_accessory_info", AsyncMock(side_effect=_enumerate)) as info:
        await getattr(connection, method)(*args)

    info.assert_awaited_once()


async def test_connect_enumerates_by_default_and_not_when_asked_not_to() -> None:
    for enumerate_database, expected in ((True, True), (False, False)):
        connection = _connection(FakeOwner(None))
        # connect() returns early on a live session; there isn't one yet.
        connection.enc_ctx = None

        with (
            patch.object(connection, "do_pair_verify", AsyncMock()),
            patch.object(connection, "get_accessory_info", AsyncMock()) as info,
        ):
            await connection.connect({}, enumerate_database=enumerate_database)

        assert bool(info.await_count) is expected
