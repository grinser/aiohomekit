"""Value pack/unpack round-trip for Pdu09Characteristic.

Covers the write path (``characteristic.value = x`` -> ``raw_value`` in
_write_characteristics_enter) and the read path (``raw_value`` -> ``value`` in
_read_characteristics_exit) for every supported format, including the precise
integer widths now emitted by to_dict(). A width mismatch would raise or corrupt
the value, so this guards the database rebuilt from a cache or signature walk.
"""

import struct

import pytest

from aiohomekit.controller.coap.structs import Pdu09Characteristic


def _char(fmt_byte, unit=0x2700):
    return Pdu09Characteristic(
        type=1,
        instance_id=1,
        properties=0x30,  # secure read + secure write
        presentation_format=struct.pack("<BxHxxx", fmt_byte, unit),
        valid_range=None,
        step_value=None,
        valid_values=None,
        valid_values_range=None,
        user_descriptor=None,
    )


@pytest.mark.parametrize(
    "fmt_byte,fmt_str,value",
    [
        (0x01, "bool", True),
        (0x01, "bool", False),
        (0x04, "uint8", 0),
        (0x04, "uint8", 255),
        (0x06, "uint16", 65535),
        (0x08, "uint32", 4294967295),
        (0x0A, "uint64", 2**63),
        (0x10, "int", -2147483648),
        (0x14, "float", 26.5),
    ],
)
def test_value_roundtrip(fmt_byte, fmt_str, value):
    # the format is reported precisely (no collapsing to "int")
    assert _char(fmt_byte).data_type_str == fmt_str

    writer = _char(fmt_byte)
    writer.value = value  # write path packs to the wire representation
    raw = writer.raw_value
    assert isinstance(raw, (bytes, bytearray))

    reader = _char(fmt_byte)
    reader.raw_value = raw  # read path unpacks back to a Python value
    if fmt_byte == 0x14:
        assert reader.value == pytest.approx(value)
    else:
        assert reader.value == value


def test_uint8_and_uint32_have_distinct_widths():
    # The bug this guards against: uint8 and uint32 both reported as "int" so a
    # 4-byte value would be unpacked as 1 byte (or vice versa).
    c8 = _char(0x04)
    c8.value = 7
    c32 = _char(0x08)
    c32.value = 7
    assert len(c8.raw_value) == 1
    assert len(c32.raw_value) == 4
