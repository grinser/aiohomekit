"""A database read that fails after a successful pair verify gets its own exception type."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aiohomekit.characteristic_cache import CharacteristicCacheMemory
from aiohomekit.controller.coap.connection import CoAPHomeKitConnection
from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.controller.controller import Controller
from aiohomekit.exceptions import AccessoryDisconnectedError, AccessoryEnumerationError


def _connection() -> CoAPHomeKitConnection:
    return CoAPHomeKitConnection(None, "any", 1234)


def _pairing() -> CoAPPairing:
    controller = Controller(char_cache=CharacteristicCacheMemory())
    return CoAPPairing(
        controller,
        {
            "AccessoryPairingID": "00:00:00:00:00:00",
            "AccessoryIP": "any",
            "AccessoryPort": 1234,
        },
    )


async def test_a_pair_verify_failure_is_not_an_enumeration_error() -> None:
    """Before pair verify has succeeded nothing is known about the database."""
    connection = _connection()

    with patch.object(connection, "do_pair_verify", AsyncMock(side_effect=RuntimeError("nope"))):
        with pytest.raises(AccessoryDisconnectedError) as excinfo:
            await connection.connect({})

    assert not isinstance(excinfo.value, AccessoryEnumerationError)


async def test_a_database_read_failure_after_pair_verify_is_typed() -> None:
    connection = _connection()
    inner = AccessoryDisconnectedError("Request timeout")

    with (
        patch.object(connection, "do_pair_verify", AsyncMock()),
        patch.object(connection, "get_accessory_info", AsyncMock(side_effect=inner)),
    ):
        with pytest.raises(AccessoryEnumerationError) as excinfo:
            await connection.connect({})

    # The cause and its message survive, so logs still say what actually failed.
    assert excinfo.value.__cause__ is inner
    assert "Request timeout" in str(excinfo.value)


async def test_a_pairing_operation_cannot_fail_enumeration() -> None:
    """connect(enumerate_database=False) never reads the database, so it can never raise for it."""
    connection = _connection()

    with (
        patch.object(connection, "do_pair_verify", AsyncMock()),
        patch.object(
            connection,
            "get_accessory_info",
            AsyncMock(side_effect=AccessoryDisconnectedError("Request timeout")),
        ) as get_accessory_info,
    ):
        await connection.connect({}, enumerate_database=False)

    get_accessory_info.assert_not_awaited()


async def test_the_connect_wrap_does_not_flatten_the_type() -> None:
    """The connect wrap must not flatten an AccessoryDisconnectedError subclass into the generic failure."""
    pairing = _pairing()
    typed = AccessoryEnumerationError("Accessory database enumeration failed")

    with patch.object(pairing.connection, "connect", AsyncMock(side_effect=typed)):
        with pytest.raises(AccessoryEnumerationError) as excinfo:
            await pairing._ensure_connected()

    assert excinfo.value is typed


async def test_the_generic_wrap_keeps_its_cause() -> None:
    pairing = _pairing()
    boom = RuntimeError("boom")

    with patch.object(pairing.connection, "connect", AsyncMock(side_effect=boom)):
        with pytest.raises(AccessoryDisconnectedError) as excinfo:
            await pairing._ensure_connected()

    assert not isinstance(excinfo.value, AccessoryEnumerationError)
    assert excinfo.value.__cause__ is boom


async def test_a_cancellation_is_not_swallowed() -> None:
    """A cancellation must not be turned into a connection error."""
    pairing = _pairing()

    with patch.object(pairing.connection, "connect", AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await pairing._ensure_connected()
