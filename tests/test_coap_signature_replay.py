"""Rebuild an accessory database from signatures captured off real hardware.

Every other test in this area encodes its signatures with the same struct the
code decodes them with, which cannot catch a wrong assumption about the wire
format. These bodies were captured over CoAP/Thread from an Eve Room, an
accessory that does not implement the 0x09 bulk read, so they exercise the
load-bearing claim behind the fallback: that a CoAP CHAR_SIG_READ response
decodes as the BLE transport's Characteristic TLV.
"""

import json
import pathlib

from aiohomekit.controller.coap.connection import (
    SIGNATURE_WALK_MAX_IID,
    SIGNATURE_WALK_MAX_MISSES,
    CoAPHomeKitConnection,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "eve_room_signatures.json"

# What the accessory really exposes, read back from the device.
EXPECTED_IIDS = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47,
    49, 50, 51, 53, 54, 56, 57, 58, 59,
]
PAIRING_SERVICE = 0x55
PAIRINGS_CHARACTERISTIC = 0x50
ACCESSORY_INFORMATION = 0x3E
# Eve's own history service: outside the HomeKit base range, so it must keep
# its full 128-bit UUID.
EVE_HISTORY_SERVICE = 0xE863F007079E48FF8F279C2605A29F52


def _signatures() -> dict[int, bytes]:
    raw = json.loads(FIXTURE.read_text())["signatures"]
    return {int(iid): bytes.fromhex(body) for iid, body in raw.items()}


def _database():
    stub = CoAPHomeKitConnection.__new__(CoAPHomeKitConnection)
    return CoAPHomeKitConnection._database_from_signatures(stub, _signatures())


def test_the_rebuilt_database_matches_the_accessory():
    database = _database()

    assert len(database.accessories) == 1, "a sensor is not a bridge"
    characteristics = [
        char for acc in database.accessories for svc in acc.services for char in svc.characteristics
    ]
    assert sorted(char.instance_id for char in characteristics) == EXPECTED_IIDS
    assert len({svc.instance_id for acc in database.accessories for svc in acc.services}) == 10


def test_base_range_types_are_shortened_but_vendor_types_are_not():
    """Signatures carry the full 128-bit UUID; lookups are written against the
    short form, so a rebuilt database has to agree with a bulk-read one. Only
    the HomeKit base range may be shortened -- this accessory also exposes a
    vendor service, which has to survive intact."""
    database = _database()

    service_types = {svc.type for acc in database.accessories for svc in acc.services}
    assert ACCESSORY_INFORMATION in service_types
    assert PAIRING_SERVICE in service_types
    assert EVE_HISTORY_SERVICE in service_types, "a vendor UUID must not be truncated"


def test_the_walk_bounds_cover_this_accessory():
    """The scan limit and the miss counter are guesses; check them against the
    one real layout available -- the largest gap here is what the miss counter
    has to survive."""
    iids = EXPECTED_IIDS
    assert max(iids) <= SIGNATURE_WALK_MAX_IID
    largest_gap = max(b - a - 1 for a, b in zip(iids, iids[1:]))
    assert largest_gap < SIGNATURE_WALK_MAX_MISSES, (
        f"a gap of {largest_gap} would end the walk early on this accessory"
    )
