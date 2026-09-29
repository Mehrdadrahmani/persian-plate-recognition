from platereader.plates import (edit_distance, normalize_plate, parse_plate, plate_tokens, to_latin,
                                to_persian_display)


def test_normalize_removes_zwj_and_maps_digits():
    assert normalize_plate("61ه‍44511") == "61ه44511"
    assert normalize_plate("۱۲ب۳۴۵۶۷") == "12ب34567"
    assert normalize_plate("12ي34567") == "12ی34567"


def test_parse_and_alef_is_one_token():
    assert parse_plate("12ب34567") == ("1234567", "ب")
    assert parse_plate("64الف73911") == ("6473911", "الف")
    assert plate_tokens("64الف73911") == ["6", "4", "الف", "7", "3", "9", "1", "1"]
    assert parse_plate("12X34567") is None and parse_plate("1ب34567") is None


def test_display_forms():
    assert to_latin("12ب34567") == "12-B-345-67"
    assert to_persian_display("12ب34567") == "۱۲ ب ۳۴۵ - ۶۷"


def test_edit_distance():
    assert edit_distance(plate_tokens("12ب34567"), plate_tokens("12ب34567")) == 0
    assert edit_distance(plate_tokens("12ب34567"), plate_tokens("13ب34567")) == 1
