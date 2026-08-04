"""CoAPPairing's pairing operations pass enumerate_database=False through to the connection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from aiohomekit.characteristic_cache import CharacteristicCacheMemory
from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.controller.controller import Controller

from .test_coap_pairing_iid import _accessories, _m2_reply

PAIRING_ID = "00:00:00:00:00:00"


def _pairing() -> CoAPPairing:
    """A real CoAPPairing whose char cache already holds the accessory map, as a controller restores it."""
    char_cache = CharacteristicCacheMemory()
    char_cache.async_create_or_update_map(PAIRING_ID, 1, _accessories().serialize())

    controller = Controller(char_cache=char_cache)
    pairing = CoAPPairing(
        controller,
        {
            "AccessoryPairingID": PAIRING_ID,
            "AccessoryIP": "any",
            "AccessoryPort": 1234,
        },
    )
    assert pairing.accessories is not None  # sanity: the cache actually loaded
    return pairing


def _install_fake_session(pairing: CoAPPairing) -> None:
    """What a successful connect() leaves behind: an enc_ctx whose post() answers with a well-formed M2."""
    m2 = _m2_reply()
    fake_enc_ctx = AsyncMock()
    fake_enc_ctx.post.return_value = (len(m2), m2)
    pairing.connection.enc_ctx = fake_enc_ctx


async def test_remove_pairing_connects_without_enumerating() -> None:
    pairing = _pairing()

    async def _connect(*args: object, **kwargs: object) -> None:
        _install_fake_session(pairing)

    with (
        patch.object(pairing.connection, "connect", AsyncMock(side_effect=_connect)) as connect,
        patch.object(pairing.connection, "get_accessory_info", AsyncMock()) as get_accessory_info,
    ):
        assert await pairing.remove_pairing("some-pairing-id") is True

    connect.assert_awaited_once_with(pairing.pairing_data, enumerate_database=False)
    get_accessory_info.assert_not_awaited()


async def test_list_pairings_connects_without_enumerating() -> None:
    pairing = _pairing()

    async def _connect(*args: object, **kwargs: object) -> None:
        _install_fake_session(pairing)

    with (
        patch.object(pairing.connection, "connect", AsyncMock(side_effect=_connect)) as connect,
        patch.object(pairing.connection, "get_accessory_info", AsyncMock()) as get_accessory_info,
    ):
        assert await pairing.list_pairings() == []

    connect.assert_awaited_once_with(pairing.pairing_data, enumerate_database=False)
    get_accessory_info.assert_not_awaited()
