import time

from parkyar import billing, jalali, plates
from parkyar.camera import Tracker
from parkyar.db import Database
from parkyar.engine import Plate


def test_plates():
    assert plates.parts("۱۲ب۳۴۵۶۷") == ("12", "ب", "345", "67")
    assert plates.parts("64الف73911") == ("64", "الف", "739", "11")
    assert not plates.is_valid("12345")
    assert plates.display("12ب34567") == "۱۲ ب ۳۴۵ - ۶۷"
    assert plates.edit_distance("12ب34567", "12ب34568") == 1
    assert plates.edit_distance("12ب34567", "12ج34567") == 1


def test_jalali():
    assert jalali.to_jalali(2026, 9, 27) == (1405, 7, 5)
    assert jalali.to_jalali(2025, 3, 21) == (1404, 1, 1)
    assert jalali.duration(135) == "۲ ساعت و ۱۵ دقیقه"
    assert jalali.duration(1500) == "۱ روز و ۱ ساعت"
    assert jalali.money(12500) == "۱۲٬۵۰۰ تومان"


def test_fee():
    t = billing.Tariff(free_minutes=10, first_hour=30000, extra_hour=20000, daily_cap=250000)
    assert billing.fee(9, t) == 0
    assert billing.fee(11, t) == 30000
    assert billing.fee(60, t) == 30000
    assert billing.fee(61, t) == 50000
    assert billing.fee(150, t) == 70000
    assert billing.fee(20 * 60, t) == 250000  # capped
    assert billing.fee(25 * 60, t) == 250000 + 20000  # second day starts again, without the first-hour price
    assert billing.fee(100, billing.Tariff(0, 10, 10, 0)) == 20


def test_sessions(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    s = db.add_entry("12ب34567", t=time.time() - 3600)
    assert db.open_session("۱۲ب۳۴۵۶۷").id == s.id
    assert [x.id for x in db.similar_open_sessions("12ب34568")] == [s.id]
    db.close_session(s.id, 30000)
    assert db.open_session("12ب34567") is None
    assert db.revenue_between(time.time() - 60, time.time() + 60) == (30000, 1)
    db.save_subscriber("99د88877", "علی", "0912", time.time() + 86400)
    assert db.add_entry("99د88877").subscriber
    db.save_settings({"capacity": 50})
    assert db.settings()["capacity"] == 50


def test_tracker():
    tr = Tracker(hits=2, window=3, cooldown=60)
    p = Plate("12ب34567", 0.9, (0, 0, 1, 1), 0.9, None)
    assert tr.update([p], now=100) == []
    assert tr.update([p], now=101) == [p]
    assert tr.update([p], now=102) == []  # cooldown
    near = Plate("12ب34568", 0.9, (0, 0, 1, 1), 0.9, None)
    assert tr.update([near], now=103) == [] and tr.update([near], now=104) == []  # one-character misread, same car
    assert tr.update([Plate("55د66677", 0.3, (0, 0, 1, 1), 0.9, None)] * 2, now=105) == []  # low confidence ignored
