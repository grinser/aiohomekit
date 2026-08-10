#
# Copyright 2022 aiohomekit team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""The signature-walk fallback is a quirk, not a capability.

A walk decides the database has ended after a run of missing instance ids.
That inference is wrong for a sparsely numbered accessory, and the 0x09
captures already in this repo prove it: a 25-miss walk would keep 26 of the
Nanoleaf bulb's 36 characteristics, 35 of the WeMo Stage's 51, and 16 of the
Schlage Encode Plus's 86. The Eve captures are numbered 2..59 and survive.

So the fallback must be unreachable for anything but the firmware that needs
it, and it must be consulted only once 0x09 has already failed -- an Eve that
answers 0x09 (test_decode_eve_energy is one) must never be diverted into it.
"""

from __future__ import annotations


import pytest

from aiohomekit.controller.coap.connection import (
    GATT_UNSUPPORTED_CONFIRMATIONS,
    CoAPHomeKitConnection,
)
from aiohomekit.controller.coap.pdu import PDUStatus
from aiohomekit.controller.coap.structs import Pdu09Database
from aiohomekit.exceptions import AccessoryDisconnectedError

from .coap_eve_harness import EVE_LAYOUT, FakeEve, build_connection

# Near-misses matter more than obvious non-matches. Everspring and Eversmart
# are real HomeKit vendors, and a bare "Eve" or an unmeasured Eve model must not
# be admitted either -- a prefix match let all of these through.
NON_EVE_MODELS = [
    "Nanoleaf Light Strip",
    "Schlage Encode Plus",
    "WeMo Stage",
    "Everspring Door Sensor",
    "Eversmart Hub",
    "EveryWare Thing",
    "Evecolor Bulb",
    "Eve",
    "EveRoom",
    "Eve Energy 20EAO8701",
]


def _walk_would_truncate(blob: bytes, max_misses: int = 25) -> tuple[int, int]:
    """(characteristics in the database, characteristics a bounded walk keeps)."""
    db = Pdu09Database.decode(blob)
    iids = sorted(c.instance_id for a in db.accessories for s in a.services for c in s.characteristics)
    present = set(iids)
    misses = 0
    stop = max(iids)
    for iid in range(1, max(iids) + 1):
        if iid in present:
            misses = 0
        else:
            misses += 1
            if misses >= max_misses:
                stop = iid
                break
    return len(iids), len([i for i in iids if i <= stop])


def test_the_repos_own_captures_are_why_this_gate_exists():
    """The evidence for the gate, kept executable so it cannot rot.

    These are real 0x09 responses contributed by other people's hardware. A
    bounded walk mangles all of them, which is what makes the fallback unsafe
    to offer to an accessory that has not been shown to need it.
    """
    from .test_coap_structs import database_nanoleaf_bulb, database_schlage_encode_plus

    for name, blob in (
        ("Nanoleaf bulb", database_nanoleaf_bulb),
        ("Schlage Encode Plus", database_schlage_encode_plus),
    ):
        total, kept = _walk_would_truncate(blob)
        assert kept < total, f"a bounded walk no longer truncates {name} ({kept}/{total})"


@pytest.mark.parametrize("model", NON_EVE_MODELS)
async def test_a_non_eve_never_walks(model: str):
    conn = build_connection(FakeEve(gatt="dropped"), model=model)

    with pytest.raises(AccessoryDisconnectedError):
        await conn.get_accessory_info()

    assert conn.info is None
    assert not conn.database_from_walk


async def test_an_accessory_with_no_zeroconf_description_never_walks():
    """No description means no model to match, and an unmatched accessory gets
    the behaviour it has today rather than a guess."""
    conn = build_connection(FakeEve(gatt="dropped"), model=None)

    with pytest.raises(AccessoryDisconnectedError):
        await conn.get_accessory_info()

    assert not conn.database_from_walk


@pytest.mark.parametrize("model", NON_EVE_MODELS)
@pytest.mark.parametrize("gatt", ["dropped", "desync"])
async def test_a_non_eve_is_never_latched_unsupported(model: str, gatt: str):
    """A timeout is not proof an accessory lacks 0x09 -- a congested mesh looks
    identical. Latching one where the verdict cannot be acted on would turn a
    single bad exchange into a permanently unreadable device."""
    conn = build_connection(FakeEve(gatt=gatt), model=model)

    with pytest.raises(AccessoryDisconnectedError):
        await conn.get_accessory_info()

    assert not conn._gatt_unsupported


@pytest.mark.parametrize(
    "status", [PDUStatus.UNSUPPORTED_PDU, PDUStatus.INVALID_REQUEST, PDUStatus.INVALID_INSTANCE_ID]
)
async def test_even_a_definitive_rejection_does_not_latch_a_non_eve(status: PDUStatus):
    conn = build_connection(FakeEve(gatt="status", gatt_status=status), model="Nanoleaf Light Strip")

    with pytest.raises(AccessoryDisconnectedError):
        await conn.get_accessory_info()

    assert not conn._gatt_unsupported


async def test_an_eve_that_answers_0x09_is_never_diverted_into_a_walk():
    """The gate is consulted only after 0x09 has failed. Eve Energy answers it
    (tests/test_controller_coap_structs.py::test_decode_eve_energy), so a
    model match alone must not send an accessory down the fallback."""
    encoded = CoAPHomeKitConnection._database_from_signatures(None, EVE_LAYOUT).encode()
    conn = build_connection(FakeEve(gatt="body", gatt_body=encoded), model="Eve Energy 20EAO8701")
    walked = False
    original = conn._signature_walk

    async def _spy(*args, **kwargs):
        nonlocal walked
        walked = True
        return await original(*args, **kwargs)

    conn._signature_walk = _spy
    await conn.get_accessory_info()

    assert not walked
    assert not conn.database_from_walk
    assert not conn._gatt_unsupported


async def test_an_eve_that_drops_0x09_still_walks():
    """The whole point. If this fails the fallback has been gated out of
    existence."""
    conn = build_connection(FakeEve(gatt="dropped"), model="Eve Room 20EBX9901")

    await conn.get_accessory_info()

    assert conn.database_from_walk
    assert conn.info is not None


async def test_a_walk_built_database_declares_no_linked_services():
    """Known and accepted, not an oversight.

    Linked services live on the service signature (0x06); the walk reads only
    characteristic signatures (0x01), so it cannot see them. That costs nothing
    on the accessories that can reach the walk: an Eve Room and an Eve Weather
    were probed with SERV_SIG_READ across every service and declared none, and
    the Eve Energy capture in this repo declares none either. The captures that
    do use linked services -- WeMo Stage and Schlage Encode Plus -- cannot
    reach the walk, because the gate keeps them on the 0x09 path.

    If a future Eve needs them, SERV_SIG_READ recovers them for about one extra
    request per service.
    """
    conn = build_connection(FakeEve(gatt="dropped"), model="Eve Room 20EBX9901")

    await conn.get_accessory_info()

    assert conn.database_from_walk
    linked = [
        service.linked_services for accessory in conn.info.accessories for service in accessory.services
    ]
    assert linked and not any(linked), "the walk cannot source linked services"


async def test_a_gated_model_is_not_condemned_by_one_unanswered_probe():
    """The gate names a product, not a firmware capability.

    A gated model whose firmware DOES implement 0x09 must survive an ambiguous
    failure -- a lost packet, a congested mesh, a sleeping accessory all look
    like silence. One of them falls back so the caller still gets a database,
    but the verdict is not recorded, so the next session probes again.
    """
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve, model="Eve Room 20EBX9901")

    await conn.get_accessory_info()

    assert conn.database_from_walk, "the caller still needs a database"
    assert not conn._gatt_unsupported, (
        "one unanswered probe latched 0x09 off; a firmware that has it would never be asked again"
    )


async def test_a_repeated_silence_does_settle_it():
    """The counterpart: the latch must still happen, or every session pays the
    full probe timeout on firmware that genuinely lacks 0x09."""
    eve = FakeEve(gatt="dropped")
    conn = build_connection(eve, model="Eve Room 20EBX9901")

    for _ in range(GATT_UNSUPPORTED_CONFIRMATIONS):
        conn.invalidate_database()
        await conn.get_accessory_info()

    assert conn._gatt_unsupported


async def test_a_config_number_change_forgets_the_verdict():
    """HAP requires the config number to change whenever the attribute database
    does, which is what a firmware update produces. The credentials the verdict
    is keyed on do not change across an update, so without this it outlives the
    firmware it was formed against for the life of the process.
    """
    conn = build_connection(FakeEve(gatt="dropped"), model="Eve Room 20EBX9901")
    conn._gatt_unsupported = True
    assert conn._gatt_unsupported

    conn.forget_gatt_verdict()

    assert not conn._gatt_unsupported, "a firmware update cannot lift the verdict"
