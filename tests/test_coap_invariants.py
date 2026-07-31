"""Invariants that must hold on every path through the CoAP transport.

These are not tests of particular call sequences. Each one states a property
the code must satisfy no matter which entry point is used, and the ids match
the verified call-graph map so a violation can be traced back to the paths that
reach it. They exist because every defect this transport has shipped came from
a path the author had not modelled, which per-path tests cannot catch.
"""

from __future__ import annotations

import asyncio

import pytest

from aiohomekit.controller.coap.connection import (
    GATT_PROBE_TIMEOUT,
    PAIRING_PROBE_TIMEOUT,
    PAIRING_SERVICE_MAX_IID,
    SIGNATURE_WALK_MAX_IID,
)
from aiohomekit.controller.coap.structs import Pdu09Database
from aiohomekit.exceptions import AccessoryDisconnectedError

from .coap_eve_harness import (
    FakeEve,
    build_connection,
    build_pairing,
)

import aiohomekit.controller.coap.connection as connection_module


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """The retry delay is real time; these tests exercise policy, not patience."""
    monkeypatch.setattr(connection_module, "PAIR_VERIFY_RETRY_DELAY", 0)


# --------------------------------------------------------------------------
# INV-1 / INV-2: what may latch _gatt_unsupported
# --------------------------------------------------------------------------


async def test_inv1_a_short_probe_may_not_latch_0x09_off():
    """A bounded read gives 0x09 a short window because the caller is in a
    hurry. Latching on that window condemns an accessory that merely answers
    slowly to a 300-request walk for the life of the connection -- which the
    constant's own comment says must not happen."""
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve)

    await conn.get_accessory_info(PAIRING_SERVICE_MAX_IID)

    assert eve.probes == [PAIRING_PROBE_TIMEOUT], "the bounded read must use the short probe"
    assert not conn._gatt_unsupported, (
        "a probe given less than GATT_PROBE_TIMEOUT proves nothing about capability"
    )


async def test_inv1_a_full_probe_may_latch_0x09_off():
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve)

    await conn.get_accessory_info()

    assert eve.probes == [GATT_PROBE_TIMEOUT]
    assert conn._gatt_unsupported


async def test_inv2_a_transient_failure_may_not_latch_0x09_off():
    """The status branch already honours this: busy or desynced is not
    "unsupported". The exception branch must agree -- a crypto desync is a
    session fault, not evidence about the accessory."""
    eve = FakeEve(gatt="desync")
    conn = build_connection(eve)

    await conn.get_accessory_info()

    assert not conn._gatt_unsupported, "a decrypt failure says nothing about 0x09 support"


# --------------------------------------------------------------------------
# INV-3: every pair-verify runs inside the caller's retry budget
# --------------------------------------------------------------------------


async def test_inv3_the_reverify_after_a_dropped_probe_uses_the_callers_budget():
    """A dropped 0x09 kills the session, so enumeration has to pair-verify
    again -- against an accessory that has just demonstrated it sleeps through
    pair-verifies. If that second verify has an implicit budget of one, the
    budget the caller asked for buys nothing on the only path that needs it."""
    eve = FakeEve(gatt="dropped")
    # Asleep for the re-verify that follows the probe, awake before and after.
    conn = build_connection(eve, sleepy_verifies=0)
    conn.enc_ctx = eve

    sleepy = {"n": 1}
    original = conn.do_pair_verify

    async def sleeps_once_after_the_probe(pairing_data):
        if eve.probes and sleepy["n"] > 0:
            sleepy["n"] -= 1
            raise asyncio.TimeoutError
        await original(pairing_data)

    conn.do_pair_verify = sleeps_once_after_the_probe

    await conn.connect(dict(), attempts=3)

    assert conn.info is not None, "the caller asked for 3 attempts; one sleepy verify must not be fatal"


# --------------------------------------------------------------------------
# INV-4 / INV-5: self.info must be usable, and never silently bounded
# --------------------------------------------------------------------------


async def test_inv4_info_is_present_after_any_successful_ensure_connected():
    """_read_characteristics_exit, _write_characteristics_enter and
    EventResource.render_put all dereference info unguarded. Any path that
    leaves a connection 'connected' with info unset turns the next poll into an
    AttributeError, which is not in the controller's catch list."""
    eve = FakeEve()
    pairing = build_pairing(eve)

    # The pairing-operation path deliberately skips enumeration...
    await pairing._ensure_connected(enumerate_database=False)
    # ...but a subsequent ordinary caller must still find a usable database.
    await pairing._ensure_connected()

    assert pairing.connection.info is not None, "a characteristic op would raise AttributeError here"


async def test_inv5_a_bounded_read_never_replaces_the_operating_database():
    """A bounded read exists to locate one characteristic. Installing its
    truncated result as connection.info makes it the lookup table for every
    later read and write."""
    eve = FakeEve()
    conn = build_connection(eve)

    await conn.get_accessory_info()
    full = conn.info
    full_iids = {c.instance_id for a in full.accessories for s in a.services for c in s.characteristics}
    assert 59 in full_iids, "precondition: the full walk sees the high sensor"

    await conn.get_accessory_info(PAIRING_SERVICE_MAX_IID)

    kept = {c.instance_id for a in conn.info.accessories for s in a.services for c in s.characteristics}
    assert 59 in kept, "a bounded read must not downgrade an already-complete database"


# --------------------------------------------------------------------------
# INV-13: an empty database must never be installed
# --------------------------------------------------------------------------


async def test_inv13_an_empty_0x09_database_is_rejected():
    """The walk path guards this; the 0x09 path did not. _pairings_characteristic
    indexes accessories[0] unguarded, so an empty database is an IndexError at
    the worst possible moment -- during an unpair."""
    empty = Pdu09Database(_accessories=[]).encode()
    eve = FakeEve(gatt="body", gatt_body=empty)
    conn = build_connection(eve)

    # Either it refuses outright or it falls back to the walk; what it must not
    # do is install a database whose accessories[0] does not exist.
    try:
        await conn.get_accessory_info()
    except AccessoryDisconnectedError:
        return
    assert conn.info is not None and list(conn.info.accessories), (
        "an empty database was installed; _pairings_characteristic indexes accessories[0]"
    )


# --------------------------------------------------------------------------
# INV-8: deadline-bound operations must not wait out a full walk
# --------------------------------------------------------------------------


async def test_inv8_remove_pairing_does_not_wait_for_a_full_enumeration():
    """The controller abandons an unpair after a few seconds, and an abandoned
    unpair orphans the pairing on the accessory. remove_pairing must therefore
    never queue behind a 300-iid walk started by someone else.

    A blocked full walk is held open for the whole test; the unpair either
    completes without it or this fails. Nothing is awaited that could hang.
    """
    eve = FakeEve()
    pairing = build_pairing(eve, session=True)
    conn = pairing.connection

    walk_started = asyncio.Event()
    hold_the_walk = asyncio.Event()
    original_walk = conn._signature_walk

    async def blocked_full_walk(max_iid=SIGNATURE_WALK_MAX_IID):
        if max_iid == SIGNATURE_WALK_MAX_IID:
            walk_started.set()
            await hold_the_walk.wait()
        return await original_walk(max_iid)

    conn._signature_walk = blocked_full_walk

    full = asyncio.create_task(conn.get_accessory_info())
    await walk_started.wait()

    removal = asyncio.create_task(conn.remove_pairing("some-controller-id"))
    try:
        await asyncio.wait_for(asyncio.shield(removal), timeout=0.25)
        blocked = False
    except asyncio.TimeoutError:
        blocked = True
    finally:
        hold_the_walk.set()
        for task in (full, removal):
            task.cancel()
        await asyncio.gather(full, removal, return_exceptions=True)

    assert not blocked, "the unpair queued behind a full walk it did not start"


# --------------------------------------------------------------------------
# Surface sweep: no public entry point may leave the connection unusable
# --------------------------------------------------------------------------


ENTRY_POINTS = [
    "get_primary_name",
    "list_accessories_and_characteristics",
    "async_populate_accessories_state",
    "get_characteristics",
    "put_characteristics",
    "subscribe",
    "unsubscribe",
    "list_pairings",
    "remove_pairing",
]


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_no_entry_point_leaves_a_connected_session_without_a_database(entry_point):
    """The sweep that would have caught the enumerate_database regression: after
    any public call, either we are not connected, or info is usable."""
    eve = FakeEve()
    pairing = build_pairing(eve)

    args: dict[str, tuple] = {
        "get_characteristics": ([(1, 2)],),
        "put_characteristics": ([(1, 2, 1)],),
        "subscribe": ([(1, 2)],),
        "unsubscribe": ([(1, 2)],),
        "remove_pairing": ("some-controller-id",),
    }
    try:
        await getattr(pairing, entry_point)(*args.get(entry_point, ()))
    except Exception:
        # Failing is allowed; leaving a live session unusable is not.
        pass

    if pairing.connection.is_connected:
        assert pairing.connection.info is not None, (
            f"{entry_point} left a connected session whose next characteristic op raises"
        )
