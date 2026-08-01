"""Invariants that must hold on every path through the CoAP transport.

These are not tests of particular call sequences. Each one states a property
the code must satisfy no matter which entry point is used, and the ids match
the verified call-graph map so a violation can be traced back to the paths that
reach it. They exist because every defect this transport has shipped came from
a path the author had not modelled, which per-path tests cannot catch.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from aiohomekit.controller.coap.connection import (
    GATT_PROBE_TIMEOUT,
    PAIRING_PROBE_TIMEOUT,
    PAIRING_SERVICE_MAX_IID,
    SIGNATURE_WALK_MAX_IID,
)
from aiohomekit.controller.coap.structs import Pdu09Database
from aiohomekit.exceptions import AccessoryDisconnectedError
from aiohomekit.protocol.tlv import HAP_TLV, TLV

from .coap_eve_harness import (
    FakeEve,
    build_connection,
    build_pairing,
    value_body,
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
    constant's own comment says must not happen.

    Driven through remove_pairing, the entry point that actually issues the
    bounded read, so the invariant is asserted against the real path."""
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve)

    assert await conn.remove_pairing("some-controller-id") is True

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


def _iids(database) -> set[int]:
    return {c.instance_id for a in database.accessories for s in a.services for c in s.characteristics}


async def test_inv5_a_bounded_read_never_replaces_the_operating_database():
    """A pairing operation must never disturb an enumerated database. The bound
    (32) sits below the highest characteristic (59), so a truncated database
    reaching self.info would be detectable as a missing iid."""
    eve = FakeEve()
    conn = build_connection(eve)

    await conn.get_accessory_info()
    assert 59 in _iids(conn.info), "precondition: the full walk sees the high sensor"
    before = conn.info

    assert await conn.remove_pairing("some-controller-id") is True

    assert conn.info is before, "a bounded read must not replace the operating database"
    assert 59 in _iids(conn.info), "a bounded read must not downgrade an already-complete database"


async def test_inv5_a_bounded_read_installs_no_database_of_its_own():
    """The same invariant from the other side: with nothing enumerated yet, a
    pairing operation must leave self.info untouched rather than installing the
    truncated database it built to find the one characteristic it needed."""
    eve = FakeEve()
    conn = build_connection(eve)

    assert await conn.remove_pairing("some-controller-id") is True

    assert conn.info is None, "the bounded read published its truncated database"
    assert max(eve.walked) <= PAIRING_SERVICE_MAX_IID, "the read must stop at the bound"


# --------------------------------------------------------------------------
# INV-13: an empty database must never be installed
# --------------------------------------------------------------------------


# Every shape of 0x09 reply that carries no usable database. The empty case is
# the interesting one: Pdu09Database(_accessories=[]).encode() is b'', and
# decoding it leaves _accessories None, so `.accessories` raises rather than
# returning [] -- an empty database cannot be constructed by decode at all.
USELESS_0X09_BODIES = {
    "empty": Pdu09Database(_accessories=[]).encode(),
    "garbage": b"\xde\xad\xbe\xef",
    "wrong-tag": bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamValue, b"\x01")])),
    "truncated-entry": bytes(TLV.encode_list([(HAP_TLV.kTLVHAPParamUnknown_18, b"")])),
}


@pytest.mark.parametrize("shape", sorted(USELESS_0X09_BODIES))
async def test_inv13_a_0x09_reply_with_no_database_is_never_installed(shape):
    """_pairings_characteristic indexes accessories[0], and the controller
    caches whatever is installed, so a database with no accessories is an
    IndexError at the worst possible moment -- during an unpair -- or a cached
    accessory that permanently has no characteristics.

    Whichever way the reply is useless, the outcome must be the same: fall back
    to the walk, or fail. Never publish it.
    """
    eve = FakeEve(gatt="body", gatt_body=USELESS_0X09_BODIES[shape])
    conn = build_connection(eve)

    try:
        await conn.get_accessory_info()
    except AccessoryDisconnectedError:
        return
    assert conn.info is not None and list(conn.info.accessories), (
        "a database with no accessories was installed"
    )
    assert not conn._gatt_unsupported, "0x09 did answer, so it must not be latched off"


async def test_inv13_a_walk_that_decodes_to_nothing_is_never_installed():
    """The same invariant on the other producer. Signatures that all fail to
    decode yield a database with no accessories, and unlike the 0x09 path
    nothing upstream raises on the way there."""
    eve = FakeEve(layout={2: b"\x00\x01\x02", 3: b"\x00\x01\x02"})
    conn = build_connection(eve)

    with pytest.raises(AccessoryDisconnectedError):
        await conn.get_accessory_info()

    assert conn.info is None, "a database with no accessories was installed"


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


async def test_inv8_connect_does_not_hold_the_connection_lock_across_a_walk():
    """connection_lock exists to keep two pair-verifies from racing; the walk
    has _enumeration_lock of its own. Holding both across the walk means any
    other connect() -- including the session-only one a pairing operation asks
    for -- waits out an enumeration it has no interest in.
    """
    eve = FakeEve()
    conn = build_connection(eve, session=False)

    walk_started = asyncio.Event()
    hold_the_walk = asyncio.Event()
    original_walk = conn._signature_walk

    async def blocked_full_walk(max_iid=SIGNATURE_WALK_MAX_IID):
        if max_iid == SIGNATURE_WALK_MAX_IID:
            walk_started.set()
            await hold_the_walk.wait()
        return await original_walk(max_iid)

    conn._signature_walk = blocked_full_walk

    enumerating = asyncio.create_task(conn.connect(dict(), enumerate_database=True))
    await walk_started.wait()

    # A second caller wanting only a session, which is already established.
    session_only = asyncio.create_task(conn.connect(dict(), enumerate_database=False))
    try:
        await asyncio.wait_for(asyncio.shield(session_only), timeout=0.25)
        blocked = False
    except asyncio.TimeoutError:
        blocked = True
    finally:
        hold_the_walk.set()
        for task in (enumerating, session_only):
            task.cancel()
        await asyncio.gather(enumerating, session_only, return_exceptions=True)

    assert not blocked, "connect() held connection_lock for the length of the walk"


async def test_inv8_remove_pairing_does_not_wait_on_a_pairing_level_enumeration():
    """The same invariant one layer up, which the connection-level fix does not
    cover. CoAPPairing funnels concurrent callers through a Condition, and the
    in-flight future used to include the enumeration -- so an unpair arriving
    while pair-verify was still running waited for the whole walk.

    That race is not hypothetical: load_pairing schedules _process_config_changed
    in the background whenever zeroconf has a cached discovery, and Home
    Assistant calls remove_pairing straight afterwards, so the two start
    together.
    """
    eve = FakeEve()
    pairing = build_pairing(eve)
    conn = pairing.connection

    verify_started = asyncio.Event()
    release_verify = asyncio.Event()
    walk_started = asyncio.Event()
    hold_the_walk = asyncio.Event()
    original_verify = conn.do_pair_verify
    original_walk = conn._signature_walk

    async def slow_verify(pairing_data):
        verify_started.set()
        await release_verify.wait()
        await original_verify(pairing_data)

    async def blocked_full_walk(max_iid=SIGNATURE_WALK_MAX_IID):
        if max_iid == SIGNATURE_WALK_MAX_IID:
            walk_started.set()
            await hold_the_walk.wait()
        return await original_walk(max_iid)

    conn.do_pair_verify = slow_verify
    conn._signature_walk = blocked_full_walk

    enumeration = asyncio.create_task(pairing.list_accessories_and_characteristics())
    await verify_started.wait()

    # The unpair arrives while the session is still being established, so it
    # cannot simply find one already there.
    removal = asyncio.create_task(pairing.remove_pairing("some-controller-id"))
    await asyncio.sleep(0)
    release_verify.set()
    await walk_started.wait()

    try:
        await asyncio.wait_for(asyncio.shield(removal), timeout=0.25)
        blocked = False
    except asyncio.TimeoutError:
        blocked = True
    finally:
        hold_the_walk.set()
        for task in (enumeration, removal):
            task.cancel()
        await asyncio.gather(enumeration, removal, return_exceptions=True)

    assert not blocked, "the unpair waited for an enumeration that was not its own"


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


ENTRY_POINT_ARGS: dict[str, tuple] = {
    "get_characteristics": ([(1, 2)],),
    "put_characteristics": ([(1, 2, 1)],),
    "subscribe": ([(1, 2)],),
    "unsubscribe": ([(1, 2)],),
    "remove_pairing": ("some-controller-id",),
}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_no_entry_point_leaves_a_session_that_cannot_be_made_usable(entry_point):
    """The sweep that would have caught the enumerate_database regression.

    Note what this does *not* say. "Connected implies info" is not the
    invariant: list_pairings and remove_pairing connect without enumerating on
    purpose, because enumerating on their behalf is what makes an unpair miss
    the controller's deadline. So a live session with no database is a legal
    state, and the property that actually protects callers is weaker and
    sufficient -- _ensure_connected, which every characteristic operation goes
    through, must always be able to produce one.
    """
    eve = FakeEve()
    pairing = build_pairing(eve)

    try:
        await getattr(pairing, entry_point)(*ENTRY_POINT_ARGS.get(entry_point, ()))
    except Exception:
        # Failing is allowed; leaving a session that cannot recover is not.
        pass

    if pairing._shutdown:
        # remove_pairing removes our own pairing and shuts the pairing down.
        # There is no later characteristic operation to protect.
        return
    if not pairing.connection.is_connected:
        return
    await pairing._ensure_connected()
    assert pairing.connection.info is not None, (
        f"after {entry_point}, _ensure_connected returned a connection whose "
        "next characteristic op raises AttributeError"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
async def test_an_event_on_an_unenumerated_session_is_not_an_attributeerror(entry_point):
    """The gap the sweep above concedes: EventResource.render_put is the one
    characteristic lookup no _ensure_connected gates, so it meets info=None for
    real on a session a pairing operation raised."""
    eve = FakeEve()
    pairing = build_pairing(eve)

    try:
        await getattr(pairing, entry_point)(*ENTRY_POINT_ARGS.get(entry_point, ()))
    except Exception:
        pass

    events: list[dict] = []
    pairing.event_received = events.append
    resource = connection_module.EventResource(pairing.connection)
    body = value_body(b"\x2a")
    payload = struct.pack("<BHH", 0, 41, len(body)) + body

    await resource.render_put(type("Request", (), {"payload": payload})())

    assert events, f"an event arriving after {entry_point} was dropped"


async def test_the_event_path_still_decodes_once_the_database_is_there():
    """Tolerating info=None must not turn into never decoding: with a database
    present the event value is the characteristic's type, not raw bytes."""
    eve = FakeEve()
    pairing = build_pairing(eve)
    await pairing._ensure_connected()

    events: list[dict] = []
    pairing.event_received = events.append
    resource = connection_module.EventResource(pairing.connection)
    body = value_body(b"\x2a")
    payload = struct.pack("<BHH", 0, 41, len(body)) + body

    await resource.render_put(type("Request", (), {"payload": payload})())

    assert events and events[0][(1, 41)]["value"] == 42, (
        "the value was reported undecoded despite a database being available"
    )
