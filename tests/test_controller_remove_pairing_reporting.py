"""A removal that does not reach the accessory must never be silent, including when it is cancelled."""

from __future__ import annotations

import asyncio
import logging

import pytest

from aiohomekit.characteristic_cache import CharacteristicCacheMemory
from aiohomekit.controller.controller import Controller
from aiohomekit.exceptions import AccessoryDisconnectedError, UnknownError


class FakePairing:
    def __init__(self, outcome: BaseException | None = None):
        self.outcome = outcome
        self.id = "aa:bb:cc:dd:ee:ff"
        self.pairing_data = {"iOSPairingId": "some-controller-id"}
        self.controller = None
        self.removed = False
        self.shutdown_called = False

    async def remove_pairing(self, pairing_id):
        if self.outcome is not None:
            raise self.outcome
        self.removed = True

    async def shutdown(self):
        self.shutdown_called = True


class FakeTransportController:
    """The transport-specific controller owning the pairing; distinct from the composite Controller under test."""

    def __init__(self):
        self.aliases: dict = {}
        self.pairings: dict = {}


def _controller(pairing: FakePairing) -> tuple[Controller, FakeTransportController]:
    controller = Controller(char_cache=CharacteristicCacheMemory())
    transport = FakeTransportController()
    pairing.controller = transport
    controller.aliases["alias"] = pairing
    controller.pairings[pairing.id] = pairing
    transport.aliases["alias"] = pairing
    transport.pairings[pairing.id] = pairing
    controller._char_cache.async_create_or_update_map(pairing.id, 1, [], None, None)
    return controller, transport


async def test_a_successful_removal_is_quiet_and_drops_the_cache():
    pairing = FakePairing()
    controller, _ = _controller(pairing)

    await controller.remove_pairing("alias")

    assert pairing.removed
    assert pairing.shutdown_called
    assert controller._char_cache.get_map(pairing.id) is None


@pytest.mark.parametrize(
    "outcome",
    [
        asyncio.CancelledError(),
        AccessoryDisconnectedError("gone"),
        RuntimeError("something else"),
    ],
    ids=["cancelled", "disconnected", "unexpected"],
)
async def test_a_failed_removal_warns_and_re_raises(outcome, caplog):
    """Every failure mode, including the one that is not an Exception."""
    pairing = FakePairing(outcome=outcome)
    controller, _ = _controller(pairing)

    with caplog.at_level(logging.WARNING, logger="aiohomekit.controller.controller"):
        with pytest.raises(type(outcome)):
            await controller.remove_pairing("alias")

    assert any("NOT removed from the accessory" in record.message for record in caplog.records), (
        f"a failed removal produced no warning; records={[r.message for r in caplog.records]}"
    )
    assert not pairing.shutdown_called, (
        "a pairing that still exists on the accessory must stay usable; shutting it "
        "down leaves the caller told to retry with nothing to retry on"
    )


async def test_a_failed_removal_keeps_the_cached_map():
    """The cached map holds the Pairings iid a retry needs, so it is kept on failure."""
    pairing = FakePairing(outcome=AccessoryDisconnectedError("gone"))
    controller, _ = _controller(pairing)

    with pytest.raises(AccessoryDisconnectedError):
        await controller.remove_pairing("alias")

    assert controller._char_cache.get_map(pairing.id) is not None, (
        "the accessory map was discarded even though the pairing still exists on the device"
    )


async def test_the_warning_names_the_accessory_and_the_remedy():
    """The warning names the device and the remedy."""
    pairing = FakePairing(outcome=asyncio.CancelledError())
    controller, _ = _controller(pairing)

    logger = logging.getLogger("aiohomekit.controller.controller")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        with pytest.raises(asyncio.CancelledError):
            await controller.remove_pairing("alias")
    finally:
        logger.removeHandler(handler)

    message = " ".join(record.getMessage() for record in records)
    assert "alias" in message
    assert "reset" in message.lower()


@pytest.mark.parametrize(
    "outcome",
    [
        asyncio.CancelledError(),
        AccessoryDisconnectedError("Remove pairing could not be confirmed"),
        UnknownError("Remove pairing failed"),
    ],
    ids=["cancelled", "unconfirmed", "rejected"],
)
async def test_a_failed_removal_leaves_the_pairing_retryable(outcome):
    """A failed removal puts the pairing back so the caller can retry it."""
    pairing = FakePairing(outcome=outcome)
    controller, _ = _controller(pairing)

    with pytest.raises(type(outcome)):
        await controller.remove_pairing("alias")

    assert controller.aliases.get("alias") is pairing, "the alias was not restored"
    assert controller.pairings.get(pairing.id) is pairing, "the pairing was not restored"
    assert not pairing.shutdown_called

    # And the retry actually works once the accessory answers.
    pairing.outcome = None
    await controller.remove_pairing("alias")

    assert pairing.removed
    assert "alias" not in controller.aliases


async def test_a_failed_removal_restores_both_controllers():
    """The restore covers both the composite Controller and the transport controller that owns the pairing."""
    pairing = FakePairing(outcome=AccessoryDisconnectedError("gone"))
    controller, transport = _controller(pairing)

    with pytest.raises(AccessoryDisconnectedError):
        await controller.remove_pairing("alias")

    assert controller.aliases.get("alias") is pairing, "the composite controller's alias was not restored"
    assert controller.pairings.get(pairing.id) is pairing, (
        "the composite controller's pairing was not restored"
    )
    assert transport.aliases.get("alias") is pairing, "the transport controller's alias was not restored"
    assert transport.pairings.get(pairing.id) is pairing, (
        "the transport controller's pairing was not restored"
    )
