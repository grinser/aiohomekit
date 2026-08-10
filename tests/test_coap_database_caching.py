"""What may be written to the persistent accessory cache, and under which config
number.

A signature walk infers the end of the database from a run of missing instance
ids. That is a guess, and on a sparsely numbered accessory it is wrong -- real
CoAP databases in this repo's fixtures run to instance id 64087 with gaps of
43515. A walk-derived database is therefore usable but never authoritative, and
must not be able to suppress a later real read -- on any path, including the
config-changed one.
"""

from aiohomekit.controller.coap.pairing import CoAPPairing
from aiohomekit.model import Accessories, AccessoriesState

ACCESSORY = [
    {
        "aid": 1,
        "services": [
            {
                "iid": 1,
                "type": "3E",
                "characteristics": [{"iid": 2, "type": "23", "perms": ["pr"], "format": "string"}],
            }
        ],
    }
]


class FakeConnection:
    address = "[::1]:5683"

    def __init__(self, partial=False, from_walk=False):
        self.database_is_partial = partial
        self.database_from_walk = from_walk
        self.invalidated = False
        self.verdict_forgotten = False

    def forget_gatt_verdict(self):
        self.verdict_forgotten = True

    async def get_accessory_info(self, verify_attempts=1):
        import copy

        self.verify_attempts = verify_attempts
        return copy.deepcopy(ACCESSORY)

    def invalidate_database(self):
        self.invalidated = True


class FakeCharCache:
    def __init__(self):
        self.persisted = []

    def async_create_or_update_map(self, pairing_id, config_num, accessories, *args, **kwargs):
        self.persisted.append(config_num)


def _pairing(prior_config_num=None, advertised_config_num=None, **kwargs):
    pairing = CoAPPairing.__new__(CoAPPairing)
    pairing.id = "AA:BB:CC:DD:EE:FF"
    pairing._accessories_state = None
    if prior_config_num is not None:
        # config_num reports -1 with no prior state, so a first read is stale by
        # construction whatever its provenance; give it a real one to tell the
        # authoritative and always-stale paths apart.
        pairing._accessories_state = AccessoriesState(Accessories(), prior_config_num)
    pairing.connection = FakeConnection(**kwargs)
    pairing.description = None
    if advertised_config_num is not None:
        pairing.description = type(
            "Description", (), {"config_num": advertised_config_num, "name": "Eve Room 4B8F"}
        )()
    pairing.controller = type("Controller", (), {"_char_cache": FakeCharCache()})()
    pairing.config_changed_listeners = set()

    # *args: _ensure_connected grows a pair_verify_attempts argument in the
    # pair-verify retry change; accept it either way so the two compose.
    async def _connected(*args, **kwargs):
        return None

    pairing._ensure_connected = _connected
    return pairing


def _persisted(pairing):
    return pairing.controller._char_cache.persisted


async def test_a_bulk_read_database_is_cached_under_the_real_config_number():
    pairing = _pairing(prior_config_num=2)

    await pairing.list_accessories_and_characteristics()

    assert _persisted(pairing) == [2], "0x09 read the whole database; it is authoritative"


async def test_a_first_bulk_read_is_not_cached_under_the_stale_marker():
    """With no prior state config_num reports -1, which is reserved for walk
    databases; an authoritative read must never be persisted under it."""
    pairing = _pairing()

    await pairing.list_accessories_and_characteristics()

    assert _persisted(pairing) == [0]
    assert pairing._accessories_state.config_num == 0


async def test_a_walk_database_is_persisted_as_always_stale_but_live_under_the_real_config():
    """Persisted under -1: a restart restores entities from it, and the first
    description update re-reads because no advertisement carries -1. In memory
    under the advertised config number: a state-number bump must keep looking
    like an event (catch-up poll), not a config change that re-walks."""
    pairing = _pairing(prior_config_num=2, advertised_config_num=5, from_walk=True)

    await pairing.list_accessories_and_characteristics()

    assert _persisted(pairing) == [-1]
    assert pairing._accessories_state.config_num == 5


async def test_a_truncated_database_is_not_cached_at_all():
    pairing = _pairing(partial=True, from_walk=True)

    await pairing.list_accessories_and_characteristics()

    assert _persisted(pairing) == [], "a database known to be cut short must not persist"
    assert pairing._accessories_state is not None, "but it stays usable in memory"
    assert pairing._accessories_state.config_num == -1, "and stays stale so the read is retried"


async def test_a_config_change_cannot_persist_a_walk_database_as_authoritative():
    """The config-changed path re-reads and then saves. For a walk database the
    save must not run: it would overwrite the always-stale marker with the
    accessory's real config number, and after a restart the walk database would
    be indistinguishable from a complete read."""
    pairing = _pairing(prior_config_num=2, advertised_config_num=7, from_walk=True)
    heard = []
    pairing.config_changed_listeners = {heard.append}

    await pairing._process_config_changed(7)

    assert pairing.connection.invalidated, "the old database must not be reused"
    assert _persisted(pairing) == [-1], "persisted only by the walk branch, never as authoritative"
    assert heard == [7], "listeners still hear the config change"


async def test_a_config_change_cannot_persist_a_truncated_database_at_all():
    pairing = _pairing(prior_config_num=2, advertised_config_num=7, from_walk=True, partial=True)
    heard = []
    pairing.config_changed_listeners = {heard.append}

    await pairing._process_config_changed(7)

    assert _persisted(pairing) == []
    assert heard == [-1], "the state stays stale, and listeners see that"


async def test_a_config_change_with_a_bulk_read_persists_normally():
    pairing = _pairing(prior_config_num=2, advertised_config_num=7)
    heard = []
    pairing.config_changed_listeners = {heard.append}

    await pairing._process_config_changed(7)

    assert _persisted(pairing) == [2, 7], "the re-read, then the config-changed save"
    assert pairing._accessories_state.config_num == 7
    assert heard == [7]


async def test_the_walk_cache_write_preserves_the_broadcast_key_and_state_number():
    """Only the config number is the always-stale sentinel.

    Passing None for the other two cleared a stored state number, which is what
    a later catch-up poll compares against to decide whether it missed events.
    """
    from aiohomekit.model import AccessoriesState

    from .coap_eve_harness import FakeEve, build_pairing

    pairing = build_pairing(FakeEve(gatt="dropped"), session=True)
    # Both live on _accessories_state, not on the description, and both default
    # to None -- so a test that does not seed them asserts None == None and
    # passes whatever the code does.
    pairing._accessories_state = AccessoriesState(pairing.accessories, pairing.config_num, b"\x01" * 32, 7)
    assert pairing.state_num is not None and pairing.broadcast_key is not None

    await pairing.list_accessories_and_characteristics()

    writes = pairing.controller._char_cache.calls
    assert writes, "the walk-built database must be cached"
    last = writes[-1]
    assert last["config_num"] == -1, "the walk's result is always stale by design"
    assert last["state_num"] == 7, f"state number was clobbered: stored {last['state_num']!r}"
    assert last["broadcast_key"] is not None, "broadcast key was clobbered"


async def test_a_config_change_forgets_what_we_believe_about_0x09():
    """Not just the database -- the capability verdict too.

    HAP requires the config number to change whenever the attribute database
    does, which is what a firmware update produces, including one that gains or
    loses the bulk read. The credentials the verdict is keyed on do not change
    across an update, so without this it outlives the firmware it was formed
    against for the life of the process.

    Asserted through _process_config_changed rather than by calling the method
    directly: the wiring is the part that was missing.
    """
    pairing = _pairing(prior_config_num=1, from_walk=True)

    await pairing._process_config_changed(9)

    assert pairing.connection.verdict_forgotten, "a config-number change left the 0x09 verdict in place"
