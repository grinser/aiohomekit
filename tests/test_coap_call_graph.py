"""The sequences a controller actually drives, and what each one may cost.

Every entry point below was read out of Home Assistant's homekit_controller
rather than assumed, because the expensive mistakes in this area have all been
wrong guesses about the caller rather than wrong protocol code. The call sites
(homeassistant/components/homekit_controller):

  pairing      config_flow.py:486  discovery.async_start_pairing(hkid)
               config_flow.py:441  finish_pairing(code)          -> CoAPPairing
               config_flow.py:582  pairing.get_primary_name()
               config_flow.py:584  pairing.close()
               config_flow.py:589  pairing.accessories_state     -> persisted by HA
  setup        connection.py:102   controller.load_pairing(unique_id, data)
               connection.py:318   async_populate_accessories_state(
                                       force_update=True, attempts=None|1)
  removal      __init__.py:111     controller.load_pairing(hkid, dict(entry.data))
               __init__.py:113     controller.remove_pairing(hkid)
  stale entry  config_flow.py:281  pairing.list_accessories_and_characteristics()

Facts that follow from that, which these tests pin down:

* `finish_pairing` builds the pairing itself, so anything it does not pass is
  absent. `AbstractPairing.description` defaults to None (abstract.py:70) and
  `AbstractPairing.__init__` never assigns it, so a description handed to the
  constructor survives -- which is how BlePairing does it (ble/pairing.py:256).
* The description cannot be delivered with `_async_description_update` at
  pairing time: it schedules `_process_config_changed` whenever the advertised
  config number exceeds ours (abstract.py:175), and ours is -1, so the pairing
  dialog would enumerate after all.
* HA never calls `list_pairings`, so only `remove_pairing` matters for the
  Pairings characteristic lookup in practice.
"""

import pytest

from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.zeroconf import HomeKitService

ADVERTISED_NAME = "Eve Room 4B8F"


def _description(config_num: int = 2) -> HomeKitService:
    return HomeKitService(
        name=ADVERTISED_NAME,
        id="fa:73:9c:4a:a2:3c",
        model="Eve Room 20EBX9901",
        feature_flags=2,
        status_flags=0,
        config_num=config_num,
        state_num=1,
        category=10,
        protocol_version="1.2",
        type="_hap._udp.local.",
        address="fdc8::1",
        addresses=["fdc8::1"],
        port=5683,
    )


PAIRING_DATA = {
    "AccessoryPairingID": "FA:73:9C:4A:A2:3C",
    "AccessoryIP": "fdc8::1",
    "AccessoryPort": 5683,
    "Connection": "CoAP",
}


class _Cache:
    def __init__(self):
        self.saved = []

    def get_map(self, pairing_id):
        return None

    def async_create_or_update_map(self, pairing_id, config_num, accessories, *a, **kw):
        self.saved.append(config_num)


def _controller():
    return type("Controller", (), {"_char_cache": _Cache()})()


def _pairing_as_finish_pairing_builds_it(description=None):
    """Construct the pairing exactly the way coap/discovery.finish_pairing does."""
    if description is None:
        return CoAPPairing(_controller(), dict(PAIRING_DATA))
    return CoAPPairing(_controller(), dict(PAIRING_DATA), description=description)


def _forbid_enumeration(pairing):
    """Enumerating during the pairing dialog is the failure being guarded."""
    calls = []

    async def enumerate_it(*args, **kwargs):
        calls.append(True)
        raise AssertionError("the pairing dialog must not enumerate the accessory")

    pairing.list_accessories_and_characteristics = enumerate_it
    return calls


async def test_the_pairing_dialog_does_not_enumerate():
    """config_flow.py:582 calls get_primary_name on the object finish_pairing
    just built. On an accessory that has to be enumerated by walking, doing so
    outlasts the dialog; the pairing is then dropped by the controller while
    the accessory keeps it, and only a factory reset clears it."""
    pairing = _pairing_as_finish_pairing_builds_it(description=_description())
    _forbid_enumeration(pairing)

    assert await pairing.get_primary_name() == ADVERTISED_NAME


async def test_finish_pairing_hands_the_description_to_the_pairing():
    """The whole dialog fix depends on it. finish_pairing constructs the
    pairing directly (coap/discovery.py), so if it does not pass the
    description there is none, get_primary_name falls through to the
    enumerating default, and the fix is silently inert -- which is exactly what
    happened on hardware. Drives the real closure rather than reading source.
    """
    from aiohomekit.controller.coap.discovery import CoAPDiscovery

    description = _description()
    controller = _controller()
    controller.pairings = {}

    discovery = CoAPDiscovery.__new__(CoAPDiscovery)
    discovery.controller = controller
    discovery.description = description

    async def fake_pair_setup(with_auth):
        return (b"salt", b"srpB")

    async def fake_pair_setup_finish(pin, salt, srpB):
        return dict(PAIRING_DATA)

    discovery.connection = type(
        "Conn",
        (),
        {"do_pair_setup": staticmethod(fake_pair_setup),
         "do_pair_setup_finish": staticmethod(fake_pair_setup_finish)},
    )()

    finish = await discovery.async_start_pairing("alias")
    pairing = await finish("123-45-678")

    assert pairing.description is description, (
        "finish_pairing must hand the advertised description to the pairing"
    )
    # And it must be usable without enumerating, which is the point of having it.
    _forbid_enumeration(pairing)
    assert await pairing.get_primary_name() == ADVERTISED_NAME


async def test_a_pairing_built_without_a_description_still_refuses_to_hang():
    """Defence in depth: if a caller ever builds a pairing with no description,
    get_primary_name has nothing advertised to return and must fall back --
    this test records that cost rather than pretending it does not exist."""
    pairing = _pairing_as_finish_pairing_builds_it()
    assert pairing.description is None

    enumerated = []

    async def enumerate_it(*args, **kwargs):
        enumerated.append(True)
        return []

    pairing.list_accessories_and_characteristics = enumerate_it
    with pytest.raises(Exception):
        await pairing.get_primary_name()
    assert enumerated, "with no advertised name there is nothing else to do"


async def test_the_state_left_for_home_assistant_to_persist_is_never_authoritative():
    """config_flow.py:589 reads accessories_state straight after
    get_primary_name and writes it to the entity map itself. Whatever the
    dialog leaves behind is therefore persisted, so it must carry a config
    number no advertisement can match, or the empty placeholder would be
    restored at setup and treated as the accessory."""
    pairing = _pairing_as_finish_pairing_builds_it(description=_description(config_num=2))
    _forbid_enumeration(pairing)

    await pairing.get_primary_name()

    state = pairing.accessories_state
    assert state.config_num == -1
    assert state.config_num != _description().config_num
    assert not list(state.accessories)


async def test_setup_repopulates_the_placeholder_the_dialog_left():
    """connection.py:318 is the only populate call for CoAP. It has to undo the
    dialog's placeholder, and `not self.accessories` cannot detect it because an
    empty Accessories() is truthy."""
    pairing = _pairing_as_finish_pairing_builds_it(description=_description(config_num=2))
    _forbid_enumeration(pairing)
    await pairing.get_primary_name()

    enumerated = []

    async def enumerate_it(*args, **kwargs):
        enumerated.append(True)

    pairing.list_accessories_and_characteristics = enumerate_it
    pairing.connection = type("Conn", (), {"database_is_partial": False})()

    await pairing.async_populate_accessories_state(force_update=True, attempts=None)

    assert enumerated, "the deferred read must happen at setup"
