"""Invariants that must hold on every path through the CoAP transport.

These are not tests of particular call sequences. Each one states a property
the code must satisfy no matter which entry point is used, and the ids match
the verified call-graph map so a violation can be traced back to the paths that
reach it. They exist because every defect this transport has shipped came from
a path the author had not modelled, which per-path tests cannot catch.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import struct

import pytest
from aiocoap.error import NetworkError as AiocoapNetworkError
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from aiohomekit.controller.ble.structs import Characteristic as CharacteristicTLV
from aiohomekit.controller.coap.connection import (
    DEFAULT_POST_TIMEOUT,
    GATT_PROBE_TIMEOUT,
    PAIRINGS_VERIFY_TIMEOUT,
    REMOVE_PAIRING_M2_TIMEOUT,
    SIGNATURE_WALK_MAX_IID,
    _PAIRING_SERVICE,
    _PAIRINGS_CHARACTERISTIC,
    _shorten_type,
)
from aiohomekit.controller.coap.structs import Pdu09Database
from aiohomekit.exceptions import AccessoryDisconnectedError
from aiohomekit.protocol.tlv import HAP_TLV, TLV

from .coap_eve_harness import (
    PAIRINGS_IID,
    FakeEve,
    bridged_cached_map,
    build_connection,
    build_pairing,
    cached_map,
    read_only_cached_map,
    real_firmware_layout,
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


# INV-1 previously had a second half: "a probe given a short window may not
# latch". Its precondition no longer exists -- the bounded pairing read that
# asked for 4 s is gone, pairing operations resolve their one iid from the
# owner's cache, and the probe takes no timeout argument for a caller to
# shorten. What remains is that the single window it does use is generous
# enough that failing it means something.


async def test_inv1_the_probe_gets_a_generous_window_before_it_latches():
    """The flag is never cleared, so a timeout here condemns the accessory to a
    300-request walk for the life of the connection. That is only defensible if
    the probe waited at least as long as any ordinary request would."""
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve)

    await conn.get_accessory_info()

    assert eve.probes == [GATT_PROBE_TIMEOUT], "the probe used something other than its own timeout"
    assert GATT_PROBE_TIMEOUT >= DEFAULT_POST_TIMEOUT, (
        "an accessory that answers a large database slowly is supported, not broken"
    )
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


async def test_inv5_a_pairing_operation_never_disturbs_the_operating_database():
    """The operating database is the lookup table for every later read and
    write. A pairing operation resolves one iid; it has no business replacing
    it."""
    eve = FakeEve()
    conn = build_connection(eve)

    await conn.get_accessory_info()
    assert 59 in _iids(conn.info), "precondition: the full walk sees the high sensor"
    before = conn.info

    assert await conn.remove_pairing("some-controller-id") is True

    assert conn.info is before, "a pairing operation replaced the operating database"
    assert 59 in _iids(conn.info), "a pairing operation downgraded a complete database"


async def test_inv5_a_cache_served_unpair_publishes_no_database():
    """Served from the owner's cache, an unpair touches no database at all --
    neither reading one nor installing one. The cached map is a model
    Accessories; self.info is a Pdu09Database, and it stays empty."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map())

    assert await pairing.remove_pairing("some-controller-id") is True

    assert pairing.connection.info is None, "a cache-served unpair published a database"


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

    Stated for the warm case, which is the only one that occurs in the field: a
    controller cannot delete a config entry without an entity map behind it. A
    cold unpair has no source for the iid but the network, so it is allowed to
    queue -- there is nothing else it could do.
    """
    eve = FakeEve()
    pairing = build_pairing(eve, session=True, cached_accessories=cached_map())
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
    pairing = build_pairing(eve, cached_accessories=cached_map())
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
# INV-12b: a dead session surfaces as a disconnect, never as AttributeError
# --------------------------------------------------------------------------


async def test_a_request_on_a_closed_session_raises_a_disconnect():
    """Observed in production during an unpair: the background enumeration a
    controller schedules raced the removal, the removal's probe tore the session
    down, and the enumeration hit `'NoneType' object has no attribute 'request'`.
    AttributeError is not in a controller's catch list, so it surfaces as an
    unhandled task error rather than a retryable disconnect."""
    key = ChaCha20Poly1305(b"\x00" * 32)
    ctx = connection_module.EncryptionContext(key, key, key, "coap://[::1]/", coap_ctx=None)

    with pytest.raises(AccessoryDisconnectedError):
        await ctx.post_bytes(b"\x00\x01\x02")


async def test_a_session_torn_down_mid_request_raises_a_disconnect():
    """The check alone is not enough: teardown does not take the context's lock,
    so a coap_ctx re-read after the await would still race. The request must be
    issued against the reference captured before it."""
    key = ChaCha20Poly1305(b"\x00" * 32)

    class Torn:
        def request(self, message):
            ctx.coap_ctx = None  # a concurrent teardown lands here

            class _Pending:
                @property
                async def response(self):
                    raise AiocoapNetworkError("gone")

            return _Pending()

        async def shutdown(self):
            return None

    ctx = connection_module.EncryptionContext(key, key, key, "coap://[::1]/", coap_ctx=Torn())

    with pytest.raises(AccessoryDisconnectedError):
        await ctx.post_bytes(b"\x00\x01\x02")


# --------------------------------------------------------------------------
# INV-17: what a pairing operation may COST
# --------------------------------------------------------------------------
#
# The category every other invariant here is missing. All of them are
# reachability predicates -- must not block, must not latch, must not publish.
# The defect that orphaned a real device violated none of them: the removal did
# exactly what INV-8 demands and still lost, because 37 correct round-trips do
# not fit in the time a controller allows.
#
# A round-trip is ~0.5 s on this hardware, and the deadline is not a constant --
# it is however long the caller's HTTP client stays connected, measured between
# 10 s and 45 s on the same device with no code change. So the requirement is
# not "be fast enough", it is "make the write the next thing on the wire".

# One signature read to confirm the cached iid, then M1 write and M2 read.
REMOVE_PAIRING_MAX_REQUESTS = 3


async def test_inv17_an_unpair_served_from_cache_costs_three_round_trips():
    """A warm pairing already holds the Pairings characteristic's instance id --
    the controller restored it from the entity map before handing us the
    pairing. Rediscovering it over the air is the whole defect."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map())

    assert await pairing.remove_pairing("some-controller-id") is True

    assert eve.probes == [], "an unpair probed 0x09"
    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID]
    assert eve.reads == [PAIRINGS_IID]
    assert eve.total_requests() <= REMOVE_PAIRING_MAX_REQUESTS, (
        f"an unpair cost {eve.total_requests()} round-trips "
        f"(~{eve.elapsed:.1f}s at {eve.rtt}s each); budget is {REMOVE_PAIRING_MAX_REQUESTS}"
    )


async def test_inv17_nothing_but_a_single_confirmation_precedes_the_commit_point():
    """Stated separately because it is the property that actually bit: the probe
    and the walk were both 'correct', and both ran before the only request that
    changes anything on the accessory.

    Exactly one request may precede the write, and it must be a signature read
    *of the cached iid* -- that is the difference between confirming a cached
    answer and searching for one. A walk would show up here as many reads, or as
    a read of an iid nobody predicted.
    """
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map())

    await pairing.remove_pairing("some-controller-id")

    assert eve.requests_before_first_write() == 1, (
        f"{eve.requests_before_first_write()} requests preceded RemovePairing M1; "
        "everything before the commit point is discovery the caller is paying for"
    )
    assert eve.walked == [PAIRINGS_IID], (
        f"expected one confirming signature read of iid {PAIRINGS_IID}, got {eve.walked}"
    )


async def test_inv17_the_cache_lookup_addresses_the_primary_accessory():
    """The wire carries no accessory id -- a write goes to a bare instance id --
    so the lookup must select accessory 1 by aid, not whichever entry happens to
    be first. A bridged map with a decoy in front makes the two differ."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=bridged_cached_map(decoy_iid=250))

    assert await pairing.remove_pairing("some-controller-id") is True

    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID], (
        "M1 went somewhere other than the primary accessory's Pairings characteristic"
    )
    assert eve.walked == [PAIRINGS_IID], (
        "the wrong accessory was chosen and the mistake cost an enumeration to undo"
    )


async def test_inv17_a_cached_characteristic_that_cannot_be_written_is_not_used():
    """A cache can be wrong about more than the instance id. Spending the one
    request that matters on a read-only characteristic is not recoverable
    inside any budget."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=read_only_cached_map())

    assert await pairing.remove_pairing("some-controller-id") is True

    assert eve.walked != [PAIRINGS_IID], (
        "a Pairings characteristic without paired_write was trusted from the cache"
    )
    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID]


async def test_inv17_an_unpair_without_a_cache_still_removes_the_pairing():
    """The fallback is allowed to be slow. It is not allowed to be wrong --
    otherwise the fast path is the only path that works and a cache miss
    silently orphans the device."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=None)

    assert await pairing.remove_pairing("some-controller-id") is True

    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID]
    assert eve.walked, "with no cache the iid has to come from somewhere"


async def test_inv17_a_stale_cached_iid_that_exists_is_not_written_to():
    """The dangerous stale case, and the reason the confirmation exists.

    iid 41 is a real characteristic on this accessory (a humidity sensor), so
    unlike an absent iid the accessory will happily accept a write to it and
    answer the follow-up read. Without the confirmation, RemovePairing goes to
    a sensor, M2 decodes, and the removal reports success -- an orphan the user
    is told did not happen.
    """
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map(pairings_iid=41))

    assert await pairing.remove_pairing("some-controller-id") is True

    assert 41 not in [iid for iid, _ in eve.writes], (
        "RemovePairing was written to a sensor characteristic the cache misnamed"
    )
    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID]


async def test_inv17_an_absent_stale_iid_never_reports_a_false_success():
    """The other stale shape: the cached iid is not on the accessory at all."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map(pairings_iid=250))

    try:
        removed = await pairing.remove_pairing("some-controller-id")
    except Exception:
        return  # failing loudly is a correct outcome

    assert removed is True
    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID], (
        "a stale cached iid was written to and the removal reported success"
    )


def test_the_confirmation_accepts_a_real_firmware_signature():
    """Signatures from a device carry the full 128-bit UUID; the constants
    compared against are the short HAP forms. The harness encodes short ints,
    so nothing else here exercises the shortening -- and dropping it would make
    the confirmation reject every real accessory, sending every unpair down the
    full-walk path this change exists to remove. Checked against the captured
    Eve Room signature for iid 18."""
    fixture = json.loads(
        (pathlib.Path(__file__).parent / "fixtures" / "eve_room_signatures.json").read_text()
    )
    sig = CharacteristicTLV.decode(bytes.fromhex(fixture["signatures"][str(PAIRINGS_IID)]))

    assert sig.type != _PAIRINGS_CHARACTERISTIC, (
        "precondition: real firmware sends the full UUID, not the short form"
    )
    assert _shorten_type(sig.type) == _PAIRINGS_CHARACTERISTIC
    assert _shorten_type(int.from_bytes(sig.service_type, "little")) == _PAIRING_SERVICE


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


async def test_inv17_the_confirmation_cannot_outlast_the_removal_budget():
    """The confirmation is the only request between the caller and the write,
    and the caller is timed by something outside this library -- a removal has
    been cancelled after as little as 10 s. post()'s 16 s default would exceed
    that budget on its own."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map())

    assert await pairing.remove_pairing("some-controller-id") is True

    assert eve.sig_timeouts == [PAIRINGS_VERIFY_TIMEOUT], (
        f"the confirmation ran under {eve.sig_timeouts}, not its own short timeout"
    )
    assert PAIRINGS_VERIFY_TIMEOUT < REMOVE_PAIRING_M2_TIMEOUT < DEFAULT_POST_TIMEOUT, (
        "the requests on the critical path must be the most tightly bounded"
    )


async def test_the_confirmation_accepts_a_real_device_signature():
    """Driven through _verify_pairings_iid itself, against signatures captured
    from hardware. The synthetic layout encodes short type ints, so code that
    forgets to shorten a 128-bit UUID looks correct under it -- while on a real
    accessory the confirmation would reject every device and send every unpair
    down the full walk."""
    eve = FakeEve(layout=real_firmware_layout())
    conn = build_connection(eve)

    assert await conn._verify_pairings_iid(PAIRINGS_IID) is True, (
        "the confirmation rejected a signature captured from a real Eve Room"
    )


async def test_an_unconfirmable_iid_is_still_used_rather_than_walked_for():
    """Silence is not evidence about the cache. Falling back to an enumeration
    here is what gets a removal abandoned; writing to a cached iid that turns
    out to be wrong fails loudly and is reported."""
    eve = FakeEve(sig_failure=AccessoryDisconnectedError)
    pairing = build_pairing(eve, cached_accessories=cached_map())

    assert await pairing.remove_pairing("some-controller-id") is True

    assert [iid for iid, _ in eve.writes] == [PAIRINGS_IID]
    assert eve.walked == [PAIRINGS_IID], (
        f"an unanswered confirmation triggered an enumeration: {eve.walked}"
    )


async def test_a_request_after_an_endpoint_change_is_a_disconnect():
    """reconnect_soon drops the session when zeroconf reports a new address.
    A task already in flight then finds enc_ctx gone. That must read as a
    retryable disconnect, not AttributeError -- which is the same shape as the
    failure observed in production one level down, in post_bytes."""
    eve = FakeEve()
    conn = build_connection(eve)
    conn.enc_ctx = eve

    await conn.reconnect_soon()

    with pytest.raises(AccessoryDisconnectedError):
        await conn.read_characteristics([(1, 2)])


async def test_a_dropped_session_leaves_no_usable_context_behind():
    """Clearing the connection's reference is not enough: anything holding the
    EncryptionContext itself -- a task waiting on its lock -- would otherwise
    go on to call request() on a Context that has been shut down, whose failure
    is not one of the two exceptions post_bytes turns into a clean disconnect."""
    eve = FakeEve()
    conn = build_connection(eve)
    conn.enc_ctx = eve
    context = conn.enc_ctx
    transport = context.coap_ctx

    await conn.reconnect_soon()

    assert context.coap_ctx is None, "the dropped session still points at its transport"
    assert transport.shutdown_calls == 1, "the transport was dropped without being shut down"


# A reviewer flagged that a cached characteristic with `perms: null` would make
# the membership test raise TypeError inside _cached_pairings_iid. It cannot:
# the model rejects it first, in Characteristic.__init__ via
# _load_accessories_from_cache, so a cache that bad fails while the pairing is
# being constructed and never reaches this lookup. Any cache that loaded has a
# list here. (That earlier failure is its own pre-existing problem -- it makes
# load_pairing raise TypeError, which a controller's removal path does not
# catch -- but it is not this transport's to fix.)
