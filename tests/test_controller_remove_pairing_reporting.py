"""A removal that does not reach the accessory must never be silent.

Controller.remove_pairing discards the local side of a pairing in `finally`
blocks that run whatever happens. So when the removal itself fails, the
accessory is left still paired -- needing a factory reset before it can be used
again -- while the controller forgets about it.

The case this is really for is cancellation. Callers remove pairings from
inside HTTP request handlers, and a client that disconnects cancels the handler
mid-flight; `CancelledError` derives from `BaseException`, so it passes through
every ordinary handler and produces no log line at all. Measured on hardware:
a removal cancelled at 10.16 s, mid-enumeration, with nothing in the log to say
the accessory was left paired.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from aiohomekit.characteristic_cache import CharacteristicCacheMemory
from aiohomekit.controller.controller import Controller
from aiohomekit.exceptions import AccessoryDisconnectedError


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


def _controller(pairing: FakePairing) -> Controller:
    controller = Controller(char_cache=CharacteristicCacheMemory())
    pairing.controller = controller
    controller.aliases["alias"] = pairing
    controller.pairings[pairing.id] = pairing
    controller._char_cache.async_create_or_update_map(pairing.id, 1, [], None, None)
    return controller


async def test_a_successful_removal_is_quiet_and_drops_the_cache():
    pairing = FakePairing()
    controller = _controller(pairing)

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
    controller = _controller(pairing)

    with caplog.at_level(logging.WARNING, logger="aiohomekit.controller.controller"):
        with pytest.raises(type(outcome)):
            await controller.remove_pairing("alias")

    assert any(
        "NOT removed from the accessory" in record.message for record in caplog.records
    ), f"a failed removal produced no warning; records={[r.message for r in caplog.records]}"
    assert pairing.shutdown_called, "the pairing must still be shut down"


async def test_a_failed_removal_keeps_the_cached_map():
    """The cache holds the Pairings characteristic's instance id, which is what
    lets a retry go straight to the write instead of re-enumerating. Discarding
    it on failure throws away the thing that makes the retry affordable."""
    pairing = FakePairing(outcome=AccessoryDisconnectedError("gone"))
    controller = _controller(pairing)

    with pytest.raises(AccessoryDisconnectedError):
        await controller.remove_pairing("alias")

    assert controller._char_cache.get_map(pairing.id) is not None, (
        "the accessory map was discarded even though the pairing still exists on the device"
    )


async def test_the_warning_names_the_accessory_and_the_remedy():
    """A user reading this needs to know which device, and that a reset is
    required -- the pairing is not recoverable from the controller side."""
    pairing = FakePairing(outcome=asyncio.CancelledError())
    controller = _controller(pairing)

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
