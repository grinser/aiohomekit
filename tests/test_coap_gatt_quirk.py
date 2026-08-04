"""0x09 must be ruled out per *accessory*, not per connection object.

Measured on hardware 2026-08-02. HA builds a fresh pairing for every removal,
and the zeroconf description that arrives with it schedules a background
_process_config_changed. On a fresh connection that enumeration had no record
that this firmware drops 0x09, so it probed: 20.002 s holding the request lock
(exactly GATT_PROBE_TIMEOUT), then the session torn down. remove_pairing's write
never went out, and the accessory was left paired while HA reported success.

remove_pairing already connects with enumerate_database=False. It was the
concurrent enumerator that had to be taught what the rest of the process
already knew.
"""

from __future__ import annotations

import pytest

from aiohomekit.controller.coap.connection import (
    GATT_PROBE_TIMEOUT,
    CoAPHomeKitConnection,
)

from .coap_eve_harness import FakeEve, build_connection, build_pairing, cached_map


def test_a_fresh_connection_inherits_the_verdict():
    """The defect, directly: what a removal's brand new pairing sees."""
    eve = FakeEve()
    first = build_pairing(eve, cached_accessories=cached_map())
    assert not first.connection._gatt_unsupported

    first.connection._gatt_unsupported = True

    second = build_pairing(eve, cached_accessories=cached_map())
    assert second.connection._gatt_unsupported, (
        "a fresh pairing re-probes 0x09 -- this is what left the accessory paired on 2026-08-02"
    )


def test_the_verdict_reaches_a_second_connection_on_the_same_pairing():
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map())
    pairing.connection._gatt_unsupported = True

    other = build_connection(eve)
    other.owner = pairing

    assert other._gatt_unsupported


def test_a_different_accessory_is_unaffected():
    """The latch is per device; it must not condemn every accessory in the
    process to a 300-request walk."""
    eve = FakeEve()
    latched = build_pairing(eve, cached_accessories=cached_map())
    latched.connection._gatt_unsupported = True

    other = build_pairing(eve, cached_accessories=cached_map())
    # AbstractPairing copies the id out of pairing_data in __init__, so the
    # attribute is what the latch keys on.
    other.id = "11:22:33:44:55:66"

    assert not other.connection._gatt_unsupported


def test_the_latch_cannot_be_cleared():
    """Nothing re-enables 0x09; a silently ignored False would hide that."""
    pairing = build_pairing(FakeEve(), cached_accessories=cached_map())

    with pytest.raises(ValueError):
        pairing.connection._gatt_unsupported = False


def test_without_an_owner_id_it_still_latches_locally():
    """A pairing assigns self.connection before super().__init__ sets self.id,
    so the id is not always knowable. That must degrade to the old
    per-connection behaviour, not to no latch at all -- which would make every
    probe repeat its timeout."""
    conn = build_connection(FakeEve())
    conn.owner = None

    assert not conn._gatt_unsupported
    conn._gatt_unsupported = True

    assert conn._gatt_unsupported
    assert not CoAPHomeKitConnection._gatt_unsupported_devices, (
        "an unidentified accessory must not be recorded process-wide"
    )


async def test_a_latched_accessory_costs_no_requests_and_no_timeout():
    """The behaviour that matters: the path that used to spend 20 s spends none."""
    eve = FakeEve()
    pairing = build_pairing(eve, cached_accessories=cached_map(), session=True)
    pairing.connection._gatt_unsupported = True
    before = eve.total_requests()

    info = await pairing.connection._probe_gatt_database()

    assert info is None, "the probe must decline, so the caller walks instead"
    assert eve.total_requests() == before, "a 0x09 went out despite the latch"
    assert eve.elapsed < GATT_PROBE_TIMEOUT


def test_the_latch_is_not_shared_between_test_runs():
    """conftest clears it; without that, ordering would decide outcomes."""
    assert not CoAPHomeKitConnection._gatt_unsupported_devices
