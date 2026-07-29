"""The pairing dialog must not enumerate the accessory.

get_primary_name() is called immediately after pair-setup. Its default reads the
whole accessory database, which on an accessory that does not implement the 0x09
bulk read means one request per instance id -- slower than a controller's pairing
dialog will wait. Failing there loses the pairing while the accessory keeps it,
leaving an orphan only a factory reset can clear.
"""

import pytest

from aiohomekit.controller.coap.pairing import CoAPPairing


def _pairing(description_name="Eve Room 4B8F"):
    pairing = CoAPPairing.__new__(CoAPPairing)
    pairing._accessories_state = None
    pairing.description = type("Description", (), {"name": description_name})()
    return pairing


async def test_primary_name_comes_from_zeroconf_without_enumerating():
    pairing = _pairing()

    async def must_not_run(*args, **kwargs):
        raise AssertionError("the pairing dialog must not enumerate the accessory")

    pairing.list_accessories_and_characteristics = must_not_run

    assert await pairing.get_primary_name() == "Eve Room 4B8F"


async def test_a_placeholder_state_is_installed_so_the_read_still_happens_later():
    """config_num -1 never matches the accessory's, so the database is always
    read for real afterwards rather than the placeholder being trusted."""
    pairing = _pairing()

    async def must_not_run(*args, **kwargs):
        raise AssertionError("must not enumerate")

    pairing.list_accessories_and_characteristics = must_not_run
    await pairing.get_primary_name()

    assert pairing._accessories_state is not None
    assert pairing._accessories_state.config_num == -1
    assert not list(pairing._accessories_state.accessories)


async def test_a_second_call_still_answers_from_zeroconf():
    """The placeholder is an empty-but-truthy Accessories(), so a naive
    `not self.accessories` guard would send the second call into the default
    implementation, which raises on an empty list instead of returning."""
    pairing = _pairing()

    async def must_not_run(*args, **kwargs):
        raise AssertionError("must not enumerate")

    pairing.list_accessories_and_characteristics = must_not_run

    assert await pairing.get_primary_name() == "Eve Room 4B8F"
    assert await pairing.get_primary_name() == "Eve Room 4B8F"


async def test_the_placeholder_does_not_satisfy_the_populate_gate():
    """async_populate_accessories_state must treat the placeholder as absent:
    if it counted as a populated database, the pairing would be set up with
    zero accessories and stay that way."""
    pairing = _pairing()
    pairing.description.config_num = 3
    pairing.connection = type("Connection", (), {"database_is_partial": False})()
    enumerated = []

    async def record(*args, **kwargs):
        enumerated.append(True)

    pairing.list_accessories_and_characteristics = record

    await pairing.get_primary_name()
    await pairing.async_populate_accessories_state()

    assert enumerated, "the deferred first read must happen at setup"


async def test_without_a_description_it_falls_back_to_enumerating():
    pairing = _pairing()
    pairing.description = None
    called = False

    async def enumerate_it(*args, **kwargs):
        nonlocal called
        called = True
        raise RuntimeError("stop here")

    pairing.list_accessories_and_characteristics = enumerate_it

    with pytest.raises(RuntimeError):
        await pairing.get_primary_name()
    assert called, "with no advertised name there is nothing else to return"
