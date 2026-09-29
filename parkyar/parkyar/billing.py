"""Parking fee: free grace period, first-hour price, price per extra (started) hour, daily cap."""
import math
from dataclasses import dataclass


@dataclass
class Tariff:
    free_minutes: int = 10
    first_hour: int = 30_000
    extra_hour: int = 20_000
    daily_cap: int = 250_000  # 0 = no cap

    @classmethod
    def from_settings(cls, s):
        return cls(int(s["free_minutes"]), int(s["first_hour"]), int(s["extra_hour"]), int(s["daily_cap"]))


def fee(minutes, t: Tariff):
    """Every started hour is charged. Each 24-hour block is capped at `daily_cap`; the first-hour price applies once."""
    if minutes <= t.free_minutes:
        return 0
    hours = max(1, math.ceil(minutes / 60))
    total, first = 0, True
    while hours > 0:
        h = min(hours, 24)
        cost = (t.first_hour + (h - 1) * t.extra_hour) if first else h * t.extra_hour
        total += min(cost, t.daily_cap) if t.daily_cap > 0 else cost
        hours -= h
        first = False
    return int(total)
