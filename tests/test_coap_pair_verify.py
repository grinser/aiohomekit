"""Retrying pair-verify so one missed exchange cannot discard a fresh pairing.

A battery-powered Thread accessory routinely sleeps through the first
pair-verify after pair-setup. If connect() gives up there, the controller
discards credentials the accessory has already committed, leaving an orphaned
pairing only a factory reset can clear.

Retrying is not free -- against an accessory that is simply not there, each
attempt holds connection_lock for another timeout -- so the budget is set by
the caller, and only an interactive setup asks for more than one attempt.
"""

import types

import pytest

import aiohomekit.controller.coap.connection as connection_module
from aiohomekit.controller.coap.connection import (
    PAIR_VERIFY_ATTEMPTS,
    CoAPHomeKitConnection,
)
from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.exceptions import AccessoryDisconnectedError, AuthenticationError

from .coap_eve_harness import fake_owner

PAIRING_DATA = {"AccessoryPairingID": "AA:BB:CC:DD:EE:FF"}


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """The retry delay is real time; nothing here needs to wait it out."""
    monkeypatch.setattr(connection_module, "PAIR_VERIFY_RETRY_DELAY", 0)


def _connection():
    owner = fake_owner()
    conn = CoAPHomeKitConnection(owner, "::1", 5683)

    async def no_enumeration(verify_attempts=1):
        return []

    # The retry policy under test ends where enumeration begins.
    conn.get_accessory_info = no_enumeration
    return conn


def _verify_that(conn, outcomes):
    """Install a fake do_pair_verify fed from a list of exceptions or None."""
    calls = []

    async def fake_verify(pairing_data):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(outcome)
        if outcome is not None:
            raise outcome
        conn.enc_ctx = types.SimpleNamespace(coap_ctx=object())

    conn.do_pair_verify = fake_verify
    return calls


async def test_connect_retries_pair_verify_when_given_a_budget():
    conn = _connection()
    calls = _verify_that(conn, [TimeoutError(), TimeoutError(), None])

    await conn.connect(PAIRING_DATA, attempts=PAIR_VERIFY_ATTEMPTS)

    assert len(calls) == 3, "the first two timeouts must not be fatal"
    assert conn.is_connected


async def test_connect_still_fails_once_the_retries_are_exhausted():
    conn = _connection()
    calls = _verify_that(conn, [TimeoutError()])

    with pytest.raises(AccessoryDisconnectedError, match="Pair verify timed out"):
        await conn.connect(PAIRING_DATA, attempts=PAIR_VERIFY_ATTEMPTS)

    assert len(calls) == PAIR_VERIFY_ATTEMPTS


async def test_connect_makes_a_single_attempt_by_default():
    """Every routine path -- polls, reads, reconnects -- takes this one. An
    accessory that is not answering must not hold connection_lock for three
    timeouts when its caller is going to try again anyway.

    Measured on a flat-battery Eve: three attempts cost ~22 s per cycle, and
    Home Assistant rebuilds the pairing on each ConfigEntryNotReady retry, so
    no per-connection "already failed" flag can ever damp this down.
    """
    conn = _connection()
    calls = _verify_that(conn, [TimeoutError()])

    with pytest.raises(AccessoryDisconnectedError):
        await conn.connect(PAIRING_DATA)

    assert len(calls) == 1


async def test_a_rejected_pairing_is_not_retried():
    """A removed pairing or a factory-reset accessory fails deterministically;
    retrying it stalls the caller for the whole backoff sequence for nothing."""
    conn = _connection()
    calls = _verify_that(conn, [AuthenticationError("no such pairing")])

    with pytest.raises(AccessoryDisconnectedError):
        await conn.connect(PAIRING_DATA, attempts=PAIR_VERIFY_ATTEMPTS)

    assert len(calls) == 1, "a deterministic rejection must fail fast"


def _pairing():
    pairing = CoAPPairing.__new__(CoAPPairing)
    pairing._accessories_state = None
    pairing.budgets = []

    async def record(pair_verify_attempts=1):
        pairing.budgets.append(pair_verify_attempts)

    pairing.list_accessories_and_characteristics = record
    return pairing


async def test_an_interactive_setup_gets_the_full_budget():
    """The controller passes attempts=None once it is running, which is when a
    device has just been paired -- the case the retry exists for."""
    pairing = _pairing()

    await pairing.async_populate_accessories_state(force_update=True, attempts=None)

    assert pairing.budgets == [PAIR_VERIFY_ATTEMPTS]


async def test_a_startup_setup_does_not_get_the_full_budget():
    """attempts=1 means the controller is still starting up and is in a hurry;
    no fresh pairing is at stake, so one attempt is right."""
    pairing = _pairing()

    await pairing.async_populate_accessories_state(force_update=True, attempts=1)

    assert pairing.budgets == [1]
