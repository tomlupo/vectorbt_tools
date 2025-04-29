# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled utilities for working with dates and time."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.datetime_ import DTCNT
from vectorbtpro.utils.formatting import prettify

__all__ = []

__pdoc__ = {}

us_ns = 1000
"""Microsecond in nanoseconds."""

ms_ns = us_ns * 1000
"""Millisecond in nanoseconds."""

s_ns = ms_ns * 1000
"""Second in nanoseconds."""

m_ns = s_ns * 60
"""Minute in nanoseconds."""

h_ns = m_ns * 60
"""Hour in nanoseconds."""

d_ns = h_ns * 24
"""Day in nanoseconds."""

w_ns = d_ns * 7
"""Week in nanoseconds."""

y_ns = (d_ns * 438291) // 1200
"""Year in nanoseconds."""

q_ns = y_ns // 4
"""Quarter in nanoseconds."""

mo_ns = q_ns // 3
"""Month in nanoseconds."""

semi_mo_ns = mo_ns // 2
"""Semi-month in nanoseconds."""

ns_td = np.timedelta64(1, "ns")
"""Nanosecond as a timedelta."""

us_td = np.timedelta64(us_ns, "ns")
"""Microsecond as a timedelta."""

ms_td = np.timedelta64(ms_ns, "ns")
"""Millisecond as a timedelta."""

s_td = np.timedelta64(s_ns, "ns")
"""Second as a timedelta."""

m_td = np.timedelta64(m_ns, "ns")
"""Minute as a timedelta."""

h_td = np.timedelta64(h_ns, "ns")
"""Hour as a timedelta."""

d_td = np.timedelta64(d_ns, "ns")
"""Day as a timedelta."""

w_td = np.timedelta64(w_ns, "ns")
"""Week as a timedelta."""

semi_mo_td = np.timedelta64(semi_mo_ns, "ns")
"""Semi-month as a timedelta."""

mo_td = np.timedelta64(mo_ns, "ns")
"""Month as a timedelta."""

q_td = np.timedelta64(q_ns, "ns")
"""Quarter as a timedelta."""

y_td = np.timedelta64(y_ns, "ns")
"""Year as a timedelta."""

unix_epoch_dt = np.datetime64(0, "ns")
"""Unix epoch (datetime)."""


@register_jitted(cache=True)
def second_remainder_nb(ts: int) -> int:
    """Get the nanosecond remainder after the second."""
    return ts % 1000000000


@register_jitted(cache=True)
def nanosecond_nb(ts: int) -> int:
    """Get the nanosecond."""
    return ts % 1000


@register_jitted(cache=True)
def microseconds_nb(ts: int) -> int:
    """Get the number of microseconds."""
    return ts // us_ns


@register_jitted(cache=True)
def microsecond_nb(ts: int) -> int:
    """Get the microsecond."""
    return microseconds_nb(ts) % (ms_ns // us_ns)


@register_jitted(cache=True)
def milliseconds_nb(ts: int) -> int:
    """Get the number of milliseconds."""
    return ts // ms_ns


@register_jitted(cache=True)
def millisecond_nb(ts: int) -> int:
    """Get the millisecond."""
    return milliseconds_nb(ts) % (s_ns // ms_ns)


@register_jitted(cache=True)
def seconds_nb(ts: int) -> int:
    """Get the number of seconds."""
    return ts // s_ns


@register_jitted(cache=True)
def second_nb(ts: int) -> int:
    """Get the seconds."""
    return seconds_nb(ts) % (m_ns // s_ns)


@register_jitted(cache=True)
def minutes_nb(ts: int) -> int:
    """Get the number of minutes."""
    return ts // m_ns


@register_jitted(cache=True)
def minute_nb(ts: int) -> int:
    """Get the minute."""
    return minutes_nb(ts) % (h_ns // m_ns)


@register_jitted(cache=True)
def hours_nb(ts: int) -> int:
    """Get the number of hours."""
    return ts // h_ns


@register_jitted(cache=True)
def hour_nb(ts: int) -> int:
    """Get the hour."""
    return hours_nb(ts) % (d_ns // h_ns)


@register_jitted(cache=True)
def days_nb(ts: int) -> int:
    """Get the number of days."""
    return ts // d_ns


@register_jitted(cache=True)
def to_civil_nb(ts: int) -> tp.Tuple[int, int, int]:
    """Convert a timestamp into a tuple of the year, month, and day."""
    z = days_nb(ts)
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return y + (m <= 2), m, d


@register_jitted(cache=True)
def from_civil_nb(y: int, m: int, d: int) -> int:
    """Convert a year, month, and day into the timestamp."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return d_ns * days


@register_jitted(cache=True)
def matches_date_nb(ts: int, y: int, m: int, d: int) -> int:
    """Check whether the timestamp match the date provided in the civil format."""
    midnight_ts1 = midnight_nb(ts)
    midnight_ts2 = from_civil_nb(y, m, d)
    return midnight_ts1 == midnight_ts2


@register_jitted(cache=True)
def day_nb(ts: int) -> int:
    """Get the day of the month."""
    y, m, d = to_civil_nb(ts)
    return d


@register_jitted(cache=True)
def midnight_nb(ts: int) -> int:
    """Get the midnight of this day."""
    return ts - ts % d_ns


@register_jitted(cache=True)
def day_changed_nb(ts1: int, ts2: int) -> bool:
    """Whether the day changed."""
    return midnight_nb(ts1) != midnight_nb(ts2)


@register_jitted(cache=True)
def weekday_from_days_nb(days: int, zero_start: bool = True) -> int:
    """Get the weekday from the total number of days.

    Weekdays are ranging from 0 (Monday) to 6 (Sunday)."""
    c_weekday = (days + 4) % 7 if days >= -4 else (days + 5) % 7 + 6
    if c_weekday == 0:
        c_weekday = 7
    if zero_start:
        c_weekday = c_weekday - 1
    return c_weekday


@register_jitted(cache=True)
def weekday_nb(ts: int, zero_start: bool = True) -> int:
    """Get the weekday.

    Weekdays are ranging from 0 (Monday) to 6 (Sunday)."""
    return weekday_from_days_nb(days_nb(ts), zero_start=zero_start)


@register_jitted(cache=True)
def weekday_diff_nb(weekday1: int, weekday2: int, zero_start: bool = True) -> int:
    """Get the difference in days between two weekdays."""
    if zero_start:
        if weekday1 > 6 or weekday1 < 0:
            raise ValueError("Weekday must be in [0, 6]")
        if weekday2 > 6 or weekday2 < 0:
            raise ValueError("Weekday must be in [0, 6]")
    else:
        if weekday1 > 7 or weekday1 < 1:
            raise ValueError("Weekday must be in [1, 7]")
        if weekday2 > 7 or weekday2 < 1:
            raise ValueError("Weekday must be in [1, 7]")
    weekday_diff = weekday1 - weekday2
    if weekday_diff <= 0:
        weekday_diff += 7
    return weekday_diff


@register_jitted(cache=True)
def past_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Get the timestamp of a weekday in the past."""
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(this_weekday, weekday, zero_start=zero_start)
    return midnight_nb(ts) - weekday_diff * d_ns


@register_jitted(cache=True)
def future_weekday_nb(ts: int, weekday: int, zero_start: bool = True) -> int:
    """Get the timestamp of a weekday in the future."""
    this_weekday = weekday_nb(ts, zero_start=zero_start)
    weekday_diff = weekday_diff_nb(weekday, this_weekday, zero_start=zero_start)
    return midnight_nb(ts) + weekday_diff * d_ns


@register_jitted(cache=True)
def day_of_year_nb(ts: int) -> int:
    """Get the day of the year."""
    y, m, d = to_civil_nb(ts)
    y_ts = from_civil_nb(y, 1, 1)
    return (ts - y_ts) // d_ns + 1


@register_jitted(cache=True)
def week_nb(ts: int) -> int:
    """Get the week of the year."""
    return day_of_year_nb(ts) // 7


@register_jitted(cache=True)
def month_nb(ts: int) -> int:
    """Get the month of the year."""
    y, m, d = to_civil_nb(ts)
    return m


@register_jitted(cache=True)
def year_nb(ts: int) -> int:
    """Get the year."""
    y, m, d = to_civil_nb(ts)
    return y


@register_jitted(cache=True)
def is_leap_year_nb(y: int) -> int:
    """Get whether the year is a leap year."""
    return (y % 4 == 0) and (y % 100 != 0 or y % 400 == 0)


@register_jitted(cache=True)
def last_day_of_month_nb(y: int, m: int) -> int:
    """Get the last day of the month."""
    if m == 1:
        return 31
    if m == 2:
        if is_leap_year_nb(y):
            return 29
        return 28
    if m == 3:
        return 31
    if m == 4:
        return 30
    if m == 5:
        return 31
    if m == 6:
        return 30
    if m == 7:
        return 31
    if m == 8:
        return 31
    if m == 9:
        return 30
    if m == 10:
        return 31
    if m == 11:
        return 30
    return 31


@register_jitted(cache=True)
def matches_dtc_nb(dtc: DTCNT, other_dtc: DTCNT) -> bool:
    """Return whether one or more datetime components match other components."""
    if dtc.year is not None and other_dtc.year is not None and dtc.year != other_dtc.year:
        return False
    if dtc.month is not None and other_dtc.month is not None and dtc.month != other_dtc.month:
        return False
    if dtc.day is not None and other_dtc.day is not None and dtc.day != other_dtc.day:
        return False
    if dtc.weekday is not None and other_dtc.weekday is not None and dtc.weekday != other_dtc.weekday:
        return False
    if dtc.hour is not None and other_dtc.hour is not None and dtc.hour != other_dtc.hour:
        return False
    if dtc.minute is not None and other_dtc.minute is not None and dtc.minute != other_dtc.minute:
        return False
    if dtc.second is not None and other_dtc.second is not None and dtc.second != other_dtc.second:
        return False
    if dtc.nanosecond is not None and other_dtc.nanosecond is not None and dtc.nanosecond != other_dtc.nanosecond:
        return False
    return True


@register_jitted(cache=True)
def index_matches_dtc_nb(index: tp.Array1d, other_dtc: DTCNT) -> tp.Array1d:
    """Run `matches_dtc_nb` on each element in an index and return a mask."""
    out = np.empty_like(index, dtype=np.bool_)
    for i in range(len(index)):
        ns = index[i]
        dtc = DTCNT(
            year=year_nb(ns),
            month=month_nb(ns),
            day=day_nb(ns),
            weekday=weekday_nb(ns),
            hour=hour_nb(ns),
            minute=minute_nb(ns),
            second=second_nb(ns),
            nanosecond=second_remainder_nb(ns),
        )
        out[i] = matches_dtc_nb(dtc, other_dtc)
    return out


class DTCST(tp.NamedTuple):
    SU: int = -3
    EU: int = -2
    U: int = -1
    O: int = 0
    I: int = 1


DTCS = DTCST()
"""_"""

__pdoc__[
    "DTCS"
] = f"""Status returned by `within_fixed_dtc_nb` and `within_periodic_dtc_nb`.

```python
{prettify(DTCS)}
```

Attributes:
    SU: Start matched, rest unknown. Move down the stack.
    EU: End matched, rest unknown. Move down the stack.
    U: Unknown. Move down the stack.
    O: Outside
    I: Inside
"""


@register_jitted(cache=True)
def within_fixed_dtc_nb(
    c: int,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
    prev_status: int = DTCS.U,
    closed_start: bool = True,
    closed_end: bool = False,
    is_last: bool = False,
) -> int:
    """Return whether a single datetime component is within a fixed range.

    Returns a status of the type `DTCS`."""
    if prev_status == DTCS.U:
        _start_c = start_c
        _end_c = end_c
    elif prev_status == DTCS.SU:
        _start_c = start_c
        _end_c = None
    elif prev_status == DTCS.EU:
        _start_c = None
        _end_c = end_c
    else:
        raise ValueError("Invalid previous DTC status")

    # Numba cannot unify int and None
    if _start_c is None:
        a = 0
    else:
        a = _start_c
    if _end_c is None:
        b = 0
    else:
        b = _end_c

    if _start_c is None and _end_c is None:
        return DTCS.U
    if _start_c is not None and _end_c is None:
        if c < a:
            return DTCS.O
        if c == a:
            if closed_start:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.SU
        if c > a:
            return DTCS.I
    if _start_c is None and _end_c is not None:
        if c < b:
            return DTCS.I
        if c == b:
            if closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.EU
        if c > b:
            return DTCS.O
    if _start_c is not None and _end_c is not None:
        if c < a or c > b:
            return DTCS.O
        if c == a and c == b:
            if closed_start and closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.U
        if c == a:
            if closed_start:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.SU
        if c == b:
            if closed_end:
                if is_last:
                    return DTCS.I
            else:
                if is_last:
                    return DTCS.O
            return DTCS.EU
        if c > a and c < b:
            return DTCS.I


@register_jitted(cache=True)
def within_periodic_dtc_nb(
    c: int,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
    prev_status: int = DTCS.U,
    closed_start: bool = True,
    closed_end: bool = False,
    overflow_later: bool = False,
    is_last: bool = False,
) -> int:
    """Return whether a single datetime component is within a periodic range.

    Returns a status of the type `DTCS`."""
    if prev_status == DTCS.U:
        _start_c = start_c
        _end_c = end_c
    elif prev_status == DTCS.SU:
        _start_c = start_c
        _end_c = None
    elif prev_status == DTCS.EU:
        _start_c = None
        _end_c = end_c
    else:
        raise ValueError("Invalid previous DTC status")

    # Numba cannot unify int and None
    if _start_c is None:
        a = 0
    else:
        a = _start_c
    if _end_c is None:
        b = 0
    else:
        b = _end_c

    if _start_c is not None and _end_c is not None and a == b:
        if overflow_later:
            return DTCS.U
    if _start_c is not None and _end_c is not None and a > b:
        status_after_start = within_fixed_dtc_nb(
            c,
            start_c=_start_c,
            end_c=None,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=is_last,
        )
        status_before_end = within_fixed_dtc_nb(
            c,
            start_c=None,
            end_c=_end_c,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=is_last,
        )
        if status_after_start == DTCS.O and status_before_end == DTCS.O:
            return DTCS.O
        if status_after_start == DTCS.I or status_before_end == DTCS.I:
            return DTCS.I
        if status_after_start == DTCS.SU:
            return DTCS.SU
        if status_before_end == DTCS.EU:
            return DTCS.EU
        return DTCS.U

    return within_fixed_dtc_nb(
        c,
        start_c=_start_c,
        end_c=_end_c,
        prev_status=prev_status,
        closed_start=closed_start,
        closed_end=closed_end,
        is_last=is_last,
    )


@register_jitted(cache=True)
def must_resolve_dtc_nb(
    c: tp.Optional[int] = None,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
) -> bool:
    """Return whether the component must be resolved."""
    if c is None:
        return False
    if start_c is None and end_c is None:
        return False
    return True


@register_jitted(cache=True)
def start_dtc_lt_nb(
    c: tp.Optional[int] = None,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
) -> bool:
    """Return whether the start component is less than the end component."""
    if c is None:
        return False
    if start_c is None:
        return False
    if end_c is None:
        return False
    return start_c < end_c


@register_jitted(cache=True)
def start_dtc_eq_nb(
    c: tp.Optional[int] = None,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
) -> bool:
    """Return whether the start component equals to the end component."""
    if c is None:
        return False
    if start_c is None:
        return False
    if end_c is None:
        return False
    return start_c == end_c


@register_jitted(cache=True)
def start_dtc_gt_nb(
    c: tp.Optional[int] = None,
    start_c: tp.Optional[int] = None,
    end_c: tp.Optional[int] = None,
) -> bool:
    """Return whether the start component is greater than the end component."""
    if c is None:
        return False
    if start_c is None:
        return False
    if end_c is None:
        return False
    return start_c > end_c


@register_jitted(cache=True)
def within_dtc_range_nb(
    dtc: DTCNT,
    start_dtc: DTCNT,
    end_dtc: DTCNT,
    closed_start: bool = True,
    closed_end: bool = False,
) -> bool:
    """Return whether one or more datetime components are within a range."""
    last = -1
    overflow_possible = True
    first_overflow = -1
    if must_resolve_dtc_nb(c=dtc.year, start_c=start_dtc.year, end_c=end_dtc.year):
        last = 0
        overflow_possible = False
    if must_resolve_dtc_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
        last = 1
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.month, start_c=start_dtc.month, end_c=end_dtc.month):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
        last = 2
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.day, start_c=start_dtc.day, end_c=end_dtc.day):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
        last = 3
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.weekday, start_c=start_dtc.weekday, end_c=end_dtc.weekday):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
        last = 4
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.hour, start_c=start_dtc.hour, end_c=end_dtc.hour):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
        last = 5
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.minute, start_c=start_dtc.minute, end_c=end_dtc.minute):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
        last = 6
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.second, start_c=start_dtc.second, end_c=end_dtc.second):
                first_overflow = last
    if must_resolve_dtc_nb(c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond):
        last = 7
        if overflow_possible and first_overflow == -1:
            if start_dtc_lt_nb(c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond):
                overflow_possible = False
        if overflow_possible and first_overflow == -1:
            if start_dtc_gt_nb(c=dtc.nanosecond, start_c=start_dtc.nanosecond, end_c=end_dtc.nanosecond):
                first_overflow = last
    if last == -1:
        return True

    prev_status = DTCS.U
    if dtc.year is not None:
        prev_status = within_fixed_dtc_nb(
            dtc.year,
            start_c=start_dtc.year,
            end_c=end_dtc.year,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            is_last=last == 0,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.month is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.month,
            start_c=start_dtc.month,
            end_c=end_dtc.month,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 1,
            is_last=last == 1,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.day is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.day,
            start_c=start_dtc.day,
            end_c=end_dtc.day,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 2,
            is_last=last == 2,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.weekday is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.weekday,
            start_c=start_dtc.weekday,
            end_c=end_dtc.weekday,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 3,
            is_last=last == 3,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.hour is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.hour,
            start_c=start_dtc.hour,
            end_c=end_dtc.hour,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 4,
            is_last=last == 4,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.minute is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.minute,
            start_c=start_dtc.minute,
            end_c=end_dtc.minute,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 5,
            is_last=last == 5,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.second is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.second,
            start_c=start_dtc.second,
            end_c=end_dtc.second,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 6,
            is_last=last == 6,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True
    if dtc.nanosecond is not None:
        prev_status = within_periodic_dtc_nb(
            dtc.nanosecond,
            start_c=start_dtc.nanosecond,
            end_c=end_dtc.nanosecond,
            prev_status=prev_status,
            closed_start=closed_start,
            closed_end=closed_end,
            overflow_later=first_overflow > 7,
            is_last=last == 7,
        )
        if prev_status == DTCS.O:
            return False
        if prev_status == DTCS.I:
            return True

    return True


@register_jitted(cache=True)
def index_within_dtc_range_nb(
    index: tp.Array1d,
    start_dtc: DTCNT,
    end_dtc: DTCNT,
    closed_start: bool = True,
    closed_end: bool = False,
) -> tp.Array1d:
    """Run `within_dtc_range_nb` on each element in an index and return a mask."""
    out = np.empty_like(index, dtype=np.bool_)
    for i in range(len(index)):
        ns = index[i]
        dtc = DTCNT(
            year=year_nb(ns),
            month=month_nb(ns),
            day=day_nb(ns),
            weekday=weekday_nb(ns),
            hour=hour_nb(ns),
            minute=minute_nb(ns),
            second=second_nb(ns),
            nanosecond=second_remainder_nb(ns),
        )
        out[i] = within_dtc_range_nb(
            dtc,
            start_dtc,
            end_dtc,
            closed_start=closed_start,
            closed_end=closed_end,
        )
    return out
