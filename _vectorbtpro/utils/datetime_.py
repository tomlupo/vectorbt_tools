# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with dates and time."""

import re
import warnings
from collections import namedtuple
from datetime import datetime, timezone, timedelta, tzinfo, date, time
from functools import partial

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset as pd_to_offset
from pandas.tseries.offsets import BaseOffset

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import min_count_nb
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, HybridConfig
from vectorbtpro.utils.parsing import WarningsFiltered

__all__ = [
    "DTC",
    "date_range",
]

__pdoc__ = {}

PandasDatetimeIndex = (pd.DatetimeIndex, pd.PeriodIndex)

# ############# Frequency ############# #


sharp_freq_str_config = HybridConfig(
    dict(
        m="m",
        M="M",
    )
)
"""_"""

__pdoc__[
    "sharp_freq_str_config"
] = f"""Config for sharp frequency mapping.

```python
{sharp_freq_str_config.prettify()}
```
"""

fuzzy_freq_str_config = HybridConfig(
    dict(
        n="ns",
        ns="ns",
        nano="ns",
        nanos="ns",
        nanosecond="ns",
        nanoseconds="ns",
        u="us",
        us="us",
        micro="us",
        micros="us",
        microsecond="us",
        microseconds="us",
        l="ms",
        ms="ms",
        milli="ms",
        millis="ms",
        millisecond="ms",
        milliseconds="ms",
        s="s",
        sec="s",
        secs="s",
        second="s",
        seconds="s",
        t="m",
        min="m",
        mins="m",
        minute="m",
        minutes="m",
        h="h",
        hour="h",
        hours="h",
        hourly="h",
        d="D",
        day="D",
        days="D",
        daily="D",
        w="W",
        wk="W",
        wks="W",
        week="W",
        weeks="W",
        weekly="W",
        mo="M",
        month="M",
        months="M",
        monthly="M",
        q="Q",
        quarter="Q",
        quarters="Q",
        quarterly="Q",
        y="Y",
        year="Y",
        years="Y",
        yearly="Y",
        annual="Y",
        annually="Y",
    )
)
"""_"""

__pdoc__[
    "fuzzy_freq_str_config"
] = f"""Config for fuzzy frequency mapping.

```python
{fuzzy_freq_str_config.prettify()}
```
"""


def split_freq_str(
    freq_str: str,
    sharp_mapping: tp.MappingLike = None,
    fuzzy_mapping: tp.MappingLike = None,
) -> tp.Optional[tp.Tuple[int, str]]:
    """Split (human-readable) frequency into multiplier and unambiguous unit.

    Can be used both as offset and timedelta.

    For mappings, see `sharp_freq_str_config` and `fuzzy_freq_str_config`.
    Sharp (case-sensitive) mappings are considered first, fuzzy (case-insensitive) mappings second.
    If a mapping returns None, will return the original unit.

    The following case-sensitive units are returned:
    * "ns" for nanosecond
    * "us" for microsecond
    * "ms" for millisecond
    * "s" for second
    * "m" for minute
    * "h" for hour
    * "D" for day
    * "W" for week
    * "M" for month
    * "Q" for quarter
    * "Y" for year

    If a unit isn't recognized, will return the original unit.
    """

    freq_str = "".join(freq_str.strip().split())
    match = re.match(r"^(\d*)\s*([a-zA-Z-]+)$", freq_str)
    if not match:
        return None
    if match.group(1) == "" and match.group(2).isnumeric():
        raise ValueError("Frequency must contain unit")
    if match.group(1) == "":
        multiplier = 1
    else:
        multiplier = int(match.group(1))
    if match.group(2) == "":
        raise ValueError("Frequency must contain unit")
    else:
        unit = match.group(2)
    if sharp_mapping is not None:
        sharp_mapping = dict(sharp_mapping)
        if unit in sharp_mapping:
            if sharp_mapping[unit] is None:
                return multiplier, unit
            return multiplier, sharp_mapping[unit]
    if unit in sharp_freq_str_config:
        if sharp_freq_str_config[unit] is None:
            return multiplier, unit
        return multiplier, sharp_freq_str_config[unit]
    if fuzzy_mapping is not None:
        fuzzy_mapping = dict(fuzzy_mapping)
        if unit.lower() in fuzzy_mapping:
            if fuzzy_mapping[unit.lower()] is None:
                return multiplier, unit
            return multiplier, fuzzy_mapping[unit.lower()]
    if unit.lower() in fuzzy_freq_str_config:
        if fuzzy_freq_str_config[unit.lower()] is None:
            return multiplier, unit
        return multiplier, fuzzy_freq_str_config[unit.lower()]
    return multiplier, unit


def prepare_offset_str(offset_str: str, allow_space: bool = False) -> str:
    """Prepare offset frequency string.

    To include multiple units, separate them with comma, semicolon, or space if `allow_space` is True.
    The output becomes comma-separated."""
    from pkg_resources import parse_version

    if parse_version(pd.__version__) < parse_version("2.2.0"):
        old_pandas = True
        year_prefix = "A"
    else:
        old_pandas = False
        year_prefix = "Y"

    if allow_space:
        freq_parts = re.split(r"[,;\s]", offset_str)
    else:
        freq_parts = re.split(r"[,;]", offset_str)
    new_freq_parts = []
    for freq_part in freq_parts:
        freq_part = " ".join(freq_part.strip().split())
        if freq_part == "":
            continue
        split = split_freq_str(freq_part, sharp_mapping=dict(MS=None))
        if split is None:
            return offset_str
        multiplier, unit = split
        if unit.lower() == "ns":
            unit = "ns"
        elif unit.lower() == "us":
            unit = "us"
        elif unit == "ms":  # case!
            unit = "ms"
        elif unit.lower() == "s":
            unit = "s"
        elif unit == "m":  # case!
            unit = "min"
        # hour
        elif unit.lower() == "h":
            unit = "h"
        elif unit.lower() in ("businesshour", "bh"):
            unit = "bh"
        elif unit.lower() in ("custombusinesshour", "cbh"):
            unit = "cbh"
        # day
        elif unit.lower() == "d":
            unit = "D"
        elif unit.lower() in ("b", "bd", "bday", "businessday"):
            unit = "B"
        elif unit.lower() in ("c", "cd", "cday", "custombusinessday"):
            unit = "C"
        # week
        elif unit.lower() in ("w", "ws", "weekstart", "weekbegin"):
            unit = "W-MON"
        elif unit.lower() in ("we", "weekend"):
            unit = "W-SUN"
        # month
        elif unit.lower() in ("m", "ms", "monthstart", "monthbegin"):
            unit = "MS"
        elif unit.lower() in ("me", "monthend"):
            if old_pandas:
                unit = "M"
            else:
                unit = "ME"
        # business month
        elif unit.lower() in (
            "bm",
            "bms",
            "bmonthstart",
            "bmonthbegin",
            "businessmonthstart",
            "businessmonthbegin",
        ):
            unit = "BMS"
        elif unit.lower() in ("bme", "bmonthend", "businessmonthend"):
            if old_pandas:
                unit = "BM"
            else:
                unit = "BME"
        # custom business month
        elif unit.lower() in (
            "cbm",
            "cbms",
            "cbmonthstart",
            "cbmonthbegin",
            "custombusinessmonthstart",
            "custombusinessmonthbegin",
        ):
            unit = "CBMS"
        elif unit.lower() in ("cbme", "cbmonthend", "custombusinessmonthend"):
            if old_pandas:
                unit = "CBM"
            else:
                unit = "CBME"
        # semi-month
        elif unit.lower() in ("sm", "sms", "semimonthstart", "semimonthbegin"):
            unit = "SMS"
        elif unit.lower() in ("sme", "semimonthend"):
            if old_pandas:
                unit = "SM"
            else:
                unit = "SME"
        # quarter
        elif unit.lower() in ("q", "qs", "quarterstart", "quarterbegin"):
            unit = "QS"
        elif unit.lower() in ("qe", "quarterend"):
            if old_pandas:
                unit = "Q"
            else:
                unit = "QE"
        # business quarter
        elif unit.lower() in (
            "bq",
            "bqs",
            "bquarterstart",
            "bquarterbegin",
            "businessquarterstart",
            "businessquarterbegin",
        ):
            unit = "BQS"
        elif unit.lower() in ("bqe", "bquarterend", "businessquarterend"):
            if old_pandas:
                unit = "BQ"
            else:
                unit = "BQE"
        # retail quarter
        elif unit.lower() in ("req", "retailquarter", "fy5253quarter"):
            unit = "REQ"
        # year
        elif unit.lower() in ("a", "y", "as", "ys", "yearstart", "yearbegin"):
            unit = year_prefix + "S"
        elif unit.lower() in ("ae", "ye", "yearend"):
            if old_pandas:
                unit = year_prefix
            else:
                unit = year_prefix + "E"
        # business year
        elif unit.lower() in (
            "ba",
            "by",
            "bas",
            "bys",
            "byearstart",
            "byearbegin",
            "businessyearstart",
            "businessyearbegin",
        ):
            unit = "B" + year_prefix + "S"
        elif unit.lower() in ("bae", "bye", "byearend", "businessyearend"):
            if old_pandas:
                unit = "B" + year_prefix
            else:
                unit = "B" + year_prefix + "E"
        # retail year
        elif unit.lower() in ("re", "retailyear", "fy5253"):
            unit = "RE"
        # day of week
        elif unit.lower() in ("mon", "monday"):
            unit = "W-MON"
        elif unit.lower() in ("tue", "tuesday"):
            unit = "W-TUE"
        elif unit.lower() in ("wed", "wednesday"):
            unit = "W-WED"
        elif unit.lower() in ("thu", "thursday"):
            unit = "W-THU"
        elif unit.lower() in ("fri", "friday"):
            unit = "W-FRI"
        elif unit.lower() in ("sat", "saturday"):
            unit = "W-SAT"
        elif unit.lower() in ("sun", "sunday"):
            unit = "W-SUN"
        # month of year
        elif unit.lower() in ("jan", "january"):
            unit = year_prefix + "S-JAN"
        elif unit.lower() in ("feb", "february"):
            unit = year_prefix + "S-FEB"
        elif unit.lower() in ("mar", "march"):
            unit = year_prefix + "S-MAR"
        elif unit.lower() in ("apr", "april"):
            unit = year_prefix + "S-APR"
        elif unit.lower() == "may":
            unit = year_prefix + "S-MAY"
        elif unit.lower() in ("jun", "june"):
            unit = year_prefix + "S-JUN"
        elif unit.lower() in ("jul", "july"):
            unit = year_prefix + "S-JUL"
        elif unit.lower() in ("aug", "august"):
            unit = year_prefix + "S-AUG"
        elif unit.lower() in ("sep", "september"):
            unit = year_prefix + "S-SEP"
        elif unit.lower() in ("oct", "october"):
            unit = year_prefix + "S-OCT"
        elif unit.lower() in ("nov", "november"):
            unit = year_prefix + "S-NOV"
        elif unit.lower() in ("dec", "december"):
            unit = year_prefix + "S-DEC"

        new_freq_parts.append(str(multiplier) + str(unit))
    return " ".join(new_freq_parts)


def to_offset(freq: tp.FrequencyLike) -> BaseOffset:
    """Convert a frequency-like object to `pd.DateOffset`."""
    if isinstance(freq, BaseOffset):
        return freq
    if isinstance(freq, str):
        freq = prepare_offset_str(freq)
    return pd_to_offset(freq)


def prepare_timedelta_str(timedelta_str: str, allow_space: bool = False) -> str:
    """Prepare timedelta frequency string.

    To include multiple units, separate them with comma, semicolon, or space if `allow_space` is True.
    The output becomes comma-separated."""
    from vectorbtpro.utils import datetime_nb as nb

    if allow_space:
        freq_parts = re.split(r"[,;\s]", timedelta_str)
    else:
        freq_parts = re.split(r"[,;]", timedelta_str)
    new_freq_parts = []
    for freq_part in freq_parts:
        freq_part = " ".join(freq_part.strip().split())
        if freq_part == "":
            continue
        split = split_freq_str(freq_part)
        if split is None:
            return timedelta_str
        multiplier, unit = split
        if unit == "m":
            unit = "min"
        elif unit == "W":
            multiplier *= 7
            unit = "D"
        elif unit == "M":
            multiplier *= nb.mo_ns / nb.d_ns
            unit = "D"
        elif unit == "Q":
            multiplier *= nb.q_ns / nb.d_ns
            unit = "D"
        elif unit == "Y":
            multiplier *= nb.y_ns / nb.d_ns
            unit = "D"
        new_freq_parts.append(str(multiplier) + str(unit))
    return " ".join(new_freq_parts)


def offset_to_timedelta(offset: BaseOffset) -> pd.Timedelta:
    """Convert offset to a timedelta."""
    from vectorbtpro.utils import datetime_nb as nb

    if isinstance(offset, (pd.offsets.BusinessHour, pd.offsets.CustomBusinessHour)):
        return pd.Timedelta(nb.h_td * offset.n)
    if isinstance(offset, (pd.offsets.BusinessDay, pd.offsets.CustomBusinessDay)):
        return pd.Timedelta(nb.d_td * offset.n)
    if isinstance(offset, pd.offsets.Week):
        return pd.Timedelta(nb.w_td * offset.n)
    if isinstance(offset, (pd.offsets.SemiMonthBegin, pd.offsets.SemiMonthEnd)):
        return pd.Timedelta(nb.semi_mo_td * offset.n)
    if isinstance(
        offset,
        (
            pd.offsets.MonthBegin,
            pd.offsets.MonthEnd,
            pd.offsets.BusinessMonthBegin,
            pd.offsets.BusinessMonthEnd,
            pd.offsets.CustomBusinessMonthBegin,
            pd.offsets.CustomBusinessMonthEnd,
            pd.offsets.WeekOfMonth,
            pd.offsets.LastWeekOfMonth,
        ),
    ):
        return pd.Timedelta(nb.mo_td * offset.n)
    if isinstance(
        offset,
        (
            pd.offsets.QuarterBegin,
            pd.offsets.QuarterEnd,
            pd.offsets.BQuarterBegin,
            pd.offsets.BQuarterEnd,
            pd.offsets.FY5253Quarter,
        ),
    ):
        return pd.Timedelta(nb.q_td * offset.n)
    if isinstance(
        offset,
        (
            pd.offsets.YearBegin,
            pd.offsets.YearEnd,
            pd.offsets.BYearBegin,
            pd.offsets.BYearEnd,
            pd.offsets.Easter,
            pd.offsets.FY5253,
        ),
    ):
        return pd.Timedelta(nb.y_td * offset.n)
    return pd.Timedelta(offset)


def fix_timedelta_precision(freq: pd.Timedelta) -> pd.Timedelta:
    """Fix the precision of timedelta."""
    if hasattr(freq, "unit") and freq.unit != "ns":
        freq = freq.as_unit("ns", round_ok=False)
    return freq


def to_timedelta(freq: tp.FrequencyLike = 1, approximate: bool = False) -> pd.Timedelta:
    """Convert a frequency-like object to `pd.Timedelta`."""
    if not isinstance(freq, pd.Timedelta):
        if isinstance(freq, str):
            freq = " ".join(freq.strip().split())
        if isinstance(freq, str) and freq.startswith("-"):
            neg_td = True
            freq = freq[1:]
        else:
            neg_td = False
        if isinstance(freq, str):
            freq = prepare_timedelta_str(freq)
        if not isinstance(freq, BaseOffset):
            try:
                if isinstance(freq, str) and not freq[0].isdigit():
                    # Otherwise "ValueError: unit abbreviation w/o a number"
                    freq = pd.Timedelta(1, unit=freq)
                else:
                    freq = pd.Timedelta(freq)
            except Exception as e1:
                try:
                    freq = to_offset(freq)
                except Exception as e2:
                    raise e1
        if isinstance(freq, BaseOffset):
            if approximate:
                freq = offset_to_timedelta(freq)
            else:
                freq = pd.Timedelta(freq)
        if neg_td:
            freq = -freq
    return fix_timedelta_precision(freq)


def to_timedelta64(freq: tp.FrequencyLike = 1) -> np.timedelta64:
    """Convert a frequency-like object to `np.timedelta64`."""
    if not isinstance(freq, np.timedelta64):
        if not isinstance(freq, pd.Timedelta):
            freq = to_timedelta(freq)
        freq = freq.to_timedelta64()
    if freq.dtype != np.dtype("timedelta64[ns]"):
        return freq.astype("timedelta64[ns]")
    return freq


def to_freq(freq: tp.FrequencyLike, allow_offset: bool = True, keep_offset: bool = False) -> tp.PandasFrequency:
    """Convert a frequency-like object to `pd.DateOffset` or `pd.Timedelta`."""
    if isinstance(freq, pd.Timedelta):
        return freq
    if allow_offset and isinstance(freq, BaseOffset):
        if not keep_offset:
            try:
                td_freq = to_timedelta(freq)
                if to_offset(td_freq) == freq:
                    freq = td_freq
                else:
                    warnings.warn(f"Ambiguous frequency {freq}", stacklevel=2)
            except Exception as e:
                pass
        return freq
    if allow_offset:
        try:
            return to_freq(to_offset(freq), allow_offset=True, keep_offset=keep_offset)
        except Exception as e:
            return to_timedelta(freq)
    return to_timedelta(freq)


# ############# Datetime ############# #


DTCNT = namedtuple("DTCNT", ["year", "month", "day", "weekday", "hour", "minute", "second", "nanosecond"])
"""Named tuple version of `DTC`."""

DTCT = tp.TypeVar("DTCT", bound="DTC")


@define
class DTC(DefineMixin):
    """Class representing one or more datetime components."""

    year: tp.Optional[int] = define.field(default=None)
    """Year."""

    month: tp.Optional[int] = define.field(default=None)
    """Month."""

    day: tp.Optional[int] = define.field(default=None)
    """Day of month."""

    weekday: tp.Optional[int] = define.field(default=None)
    """Day of week."""

    hour: tp.Optional[int] = define.field(default=None)
    """Hour."""

    minute: tp.Optional[int] = define.field(default=None)
    """Minute."""

    second: tp.Optional[int] = define.field(default=None)
    """Second."""

    nanosecond: tp.Optional[int] = define.field(default=None)
    """Nanosecond."""

    @classmethod
    def from_datetime(cls: tp.Type[DTCT], dt: tp.Datetime) -> DTCT:
        """Get `DTC` instance from a `datetime.datetime` object."""
        if isinstance(dt, np.datetime64):
            dt = pd.Timestamp(dt)
        if isinstance(dt, pd.Timestamp):
            nanosecond = dt.microsecond * 1000 + dt.nanosecond
            dt = dt.to_pydatetime(warn=False)
        else:
            nanosecond = dt.microsecond * 1000
        return cls(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            weekday=dt.weekday(),
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            nanosecond=nanosecond,
        )

    @classmethod
    def from_date(cls: tp.Type[DTCT], d: date) -> DTCT:
        """Get `DTC` instance from a `datetime.date` object."""
        return cls(year=d.year, month=d.month, day=d.day, weekday=d.weekday())

    @classmethod
    def from_time(cls: tp.Type[DTCT], t: time) -> DTCT:
        """Get `DTC` instance from a `datetime.time` object."""
        return cls(hour=t.hour, minute=t.minute, second=t.second, nanosecond=t.microsecond * 1000)

    @classmethod
    def parse_time_str(cls: tp.Type[DTCT], time_str: str, **parse_kwargs) -> DTCT:
        """Parse `DTC` instance from a time string."""
        from dateutil.parser import parser

        result = parser()._parse(time_str, **parse_kwargs)[0]
        if result.microsecond is None:
            nanosecond = None
        else:
            nanosecond = result.microsecond * 1000
        return cls(
            year=result.year,
            month=result.month,
            day=result.day,
            weekday=result.weekday,
            hour=result.hour,
            minute=result.minute,
            second=result.second,
            nanosecond=nanosecond,
        )

    @classmethod
    def from_namedtuple(cls: tp.Type[DTCT], dtc: DTCNT) -> DTCT:
        """Get `DTC` instance from a named tuple of the type `DTCNT`."""
        return cls(
            year=dtc.year,
            month=dtc.month,
            day=dtc.day,
            weekday=dtc.weekday,
            hour=dtc.hour,
            minute=dtc.minute,
            second=dtc.second,
            nanosecond=dtc.nanosecond,
        )

    @classmethod
    def parse(cls: tp.Type[DTCT], dtc: tp.DTCLike, **parse_kwargs) -> DTCT:
        """Parse `DTC` instance from a datetime-component-like object."""
        if checks.is_namedtuple(dtc):
            return cls.from_namedtuple(dtc)
        if isinstance(dtc, np.datetime64):
            dtc = pd.Timestamp(dtc)
        if isinstance(dtc, pd.Timestamp):
            if dtc.tzinfo is not None:
                raise ValueError("DTC doesn't support timezones")
            dtc = dtc.to_pydatetime()
        if isinstance(dtc, datetime):
            if dtc.tzinfo is not None:
                raise ValueError("DTC doesn't support timezones")
            return cls.from_datetime(dtc)
        if isinstance(dtc, date):
            return cls.from_date(dtc)
        if isinstance(dtc, time):
            return cls.from_time(dtc)
        if isinstance(dtc, (int, str)):
            return cls.parse_time_str(str(dtc), **parse_kwargs)
        raise TypeError(f"Invalid type {type(dtc)}")

    @classmethod
    def is_parsable(
        cls: tp.Type[DTCT],
        dtc: tp.DTCLike,
        check_func: tp.Optional[tp.Callable] = None,
        **parse_kwargs,
    ) -> bool:
        """Check whether a datetime-component-like object is parsable."""
        try:
            if isinstance(dtc, DTC):
                return True
            dtc = cls.parse(dtc, **parse_kwargs)
            if check_func is not None and not check_func(dtc):
                return False
            return True
        except Exception as e:
            pass
        return False

    def has_date(self) -> bool:
        """Whether any date component is set."""
        return self.year is not None or self.month is not None or self.day is not None

    def has_full_date(self) -> bool:
        """Whether all date components are set."""
        return self.year is not None and self.month is not None and self.day is not None

    def has_weekday(self) -> bool:
        """Whether the weekday component is set."""
        return self.weekday is not None

    def has_time(self) -> bool:
        """Whether any time component is set."""
        return (
            self.hour is not None or self.minute is not None or self.second is not None or self.nanosecond is not None
        )

    def has_full_time(self) -> bool:
        """Whether all time components are set."""
        return (
            self.hour is not None
            and self.minute is not None
            and self.second is not None
            and self.nanosecond is not None
        )

    def has_full_datetime(self) -> bool:
        """Whether all components are set."""
        return self.has_full_date() and self.has_full_time()

    def is_not_none(self) -> bool:
        """Check whether any component is set."""
        return self.has_date() or self.has_weekday() or self.has_time()

    def to_time(self) -> time:
        """Convert to a `datetime.time` instance.

        Fields that are None will become 0."""
        return time(
            hour=self.hour if self.hour is not None else 0,
            minute=self.minute if self.minute is not None else 0,
            second=self.second if self.second is not None else 0,
            microsecond=self.nanosecond // 1000 if self.nanosecond is not None else 0,
        )

    def to_namedtuple(self) -> namedtuple:
        """Convert to a named tuple."""
        return DTCNT(*self.asdict().values())


def time_to_timedelta(t: tp.Union[tp.TimeLike, DTC], **kwargs) -> pd.Timedelta:
    """Convert a time-like object into `pd.Timedelta`."""
    if isinstance(t, str):
        t = DTC.parse_time_str(t, **kwargs)
    if isinstance(t, DTC):
        if t.has_date():
            raise ValueError("Time string has a date component")
        if t.has_weekday():
            raise ValueError("Time string has a weekday component")
        if not t.has_time():
            raise ValueError("Time string doesn't have a time component")
        t = t.to_time()

    return pd.Timedelta(
        hours=t.hour if t.hour is not None else 0,
        minutes=t.minute if t.minute is not None else 0,
        seconds=t.second if t.second is not None else 0,
        milliseconds=(t.microsecond // 1000) if t.microsecond is not None else 0,
        microseconds=(t.microsecond % 1000) if t.microsecond is not None else 0,
    )


def get_utc_tz(**kwargs) -> tzinfo:
    """Get UTC timezone."""
    from dateutil.tz import tzutc

    return to_timezone(tzutc(), **kwargs)


def get_local_tz(**kwargs) -> tzinfo:
    """Get local timezone."""
    from dateutil.tz import tzlocal

    return to_timezone(tzlocal(), **kwargs)


def convert_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).timetz()


def tzaware_to_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def naive_to_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time().replace(tzinfo=tz_out)


def convert_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def is_tz_aware(dt: tp.Union[datetime, pd.Timestamp, pd.DatetimeIndex]) -> bool:
    """Whether datetime is timezone-aware."""
    tz = dt.tzinfo
    if tz is None:
        return False
    return tz.utcoffset(datetime.now()) is not None


def to_timezone(
    tz: tp.Optional[tp.TimezoneLike] = None,
    to_fixed_offset: tp.Optional[bool] = None,
    parse_with_dateparser: tp.Optional[bool] = None,
    dateparser_kwargs: tp.KwargsLike = None,
) -> tzinfo:
    """Parse the timezone.

    If the object is None, returns the local timezone. If a string, will parse with Pandas and
    dateparser (`parse_with_dateparser` must be True).

    If `to_fixed_offset` is set to True, will convert to `datetime.timezone`. See global settings.

    `dateparser_kwargs` are passed to `dateparser.parse`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if tz is None:
        return get_local_tz()
    if to_fixed_offset is None:
        to_fixed_offset = datetime_cfg["to_fixed_offset"]
    if parse_with_dateparser is None:
        parse_with_dateparser = datetime_cfg["parse_with_dateparser"]
    dateparser_kwargs = merge_dicts(datetime_cfg["dateparser_kwargs"], dateparser_kwargs)

    if isinstance(tz, str):
        try:
            tz = pd.Timestamp("now", tz=tz).tz
        except Exception as e:
            if parse_with_dateparser:
                try:
                    dt = dateparser.parse("now %s" % tz, **dateparser_kwargs)
                    if dt is not None:
                        tz = dt.tzinfo
                        to_fixed_offset = True
                except Exception as e:
                    pass
    if checks.is_number(tz):
        tz = timezone(timedelta(seconds=tz))
    if isinstance(tz, timedelta):
        tz = timezone(tz)
    if isinstance(tz, tzinfo):
        if to_fixed_offset is None:
            if tz == tz:
                to_fixed_offset = False
            else:  # Pandas has issues with this
                to_fixed_offset = True
        if to_fixed_offset:
            return timezone(tz.utcoffset(datetime.now()))
        return tz
    raise ValueError(f"Could not parse the timezone {tz}")


def to_timestamp(
    dt: tp.DatetimeLike = "now",
    parse_with_dateparser: tp.Optional[bool] = None,
    dateparser_kwargs: tp.KwargsLike = None,
    unit: str = "ns",
    tz: tp.TimezoneLike = None,
    to_fixed_offset: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Optional[pd.Timestamp]:
    """Parse the datetime as a `pd.Timestamp`.

    If the object is a string, will parse with Pandas and dateparser (`parse_with_dateparser` must be True).

    `dateparser_kwargs` are passed to `dateparser.parse` while `**kwargs` are passed to `pd.Timestamp`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if parse_with_dateparser is None:
        parse_with_dateparser = datetime_cfg["parse_with_dateparser"]
    dateparser_kwargs = merge_dicts(datetime_cfg["dateparser_kwargs"], dateparser_kwargs)
    if tz is not None:
        tz = to_timezone(
            tz,
            to_fixed_offset=to_fixed_offset,
            parse_with_dateparser=parse_with_dateparser,
            dateparser_kwargs=dateparser_kwargs,
        )

    if checks.is_number(dt):
        dt = pd.Timestamp(dt, tz="utc", unit=unit, **kwargs)
    elif isinstance(dt, str):
        dt = " ".join(dt.strip().split())
        try:
            tz = to_timezone(
                dt.split(" ")[-1],
                to_fixed_offset=to_fixed_offset,
                parse_with_dateparser=parse_with_dateparser,
                dateparser_kwargs=dateparser_kwargs,
            )
            dt = " ".join(dt.split(" ")[:-1])
        except Exception as e:
            pass
        if dt.lower() == "now":
            dt = pd.Timestamp.now(tz=tz)
        elif dt.lower() == "today":
            dt = pd.Timestamp.now(tz=tz).floor("1D")
        elif dt.lower() == "yesterday":
            dt = pd.Timestamp.now(tz=tz).floor("1D") - pd.Timedelta(days=1)
        elif dt.lower() == "tomorrow":
            dt = pd.Timestamp.now(tz=tz).floor("1D") + pd.Timedelta(days=1)
        else:
            try:
                dt = pd.Timestamp(dt, **kwargs)
            except Exception as e:
                if parse_with_dateparser:
                    try:
                        import dateparser

                        settings = dateparser_kwargs.get("settings", {})
                        settings["RELATIVE_BASE"] = settings.get(
                            "RELATIVE_BASE",
                            pd.Timestamp.now(tz=tz).to_pydatetime(),
                        )
                        dateparser_kwargs["settings"] = settings
                        dt = dateparser.parse(dt, **dateparser_kwargs)
                        if dt is not None:
                            if is_tz_aware(dt):
                                tz = to_timezone(
                                    dt.tzinfo,
                                    to_fixed_offset=True,
                                    parse_with_dateparser=parse_with_dateparser,
                                    dateparser_kwargs=dateparser_kwargs,
                                )
                                dt = dt.replace(tzinfo=tz)
                            dt = pd.Timestamp(dt, **kwargs)
                        else:
                            raise ValueError(f"Could not parse the timestamp {dt}")
                    except Exception as e:
                        raise ValueError(f"Could not parse the timestamp {dt}")
                else:
                    raise ValueError(f"Could not parse the timestamp {dt}")
    elif not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt, **kwargs)
    if tz is not None:
        if not is_tz_aware(dt):
            dt = dt.tz_localize(tz)
        else:
            dt = dt.tz_convert(tz)
    return dt


to_local_timestamp = partial(to_timestamp, tz="tzlocal()")
"""Alias for `to_timestamp` with `tz="tzlocal()"`."""

to_utc_timestamp = partial(to_timestamp, tz="utc")
"""Alias for `to_timestamp` with `tz="utc"`."""


def to_tzaware_timestamp(
    dt: tp.DatetimeLike = "now",
    naive_tz: tp.TimezoneLike = None,
    tz: tp.TimezoneLike = None,
    **kwargs,
) -> pd.Timestamp:
    """Parse the datetime as a timezone-aware `pd.Timestamp`.

    Uses `to_timestamp`.

    Raw timestamps are localized to UTC, while naive datetime is localized to `naive_tz`.
    Set `naive_tz` to None to use the default value defined under `vectorbtpro._settings.datetime`.
    To explicitly convert the datetime to a timezone, use `tz` (uses `to_timezone`).

    For defaults, see `vectorbtpro._settings.datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if naive_tz is None:
        naive_tz = datetime_cfg["naive_tz"]

    ts = to_timestamp(dt, tz=naive_tz, **kwargs)
    if is_tz_aware(ts):
        ts = ts.tz_localize(None).tz_localize(to_timezone(ts.tzinfo))
    if tz is not None:
        ts = ts.tz_convert(to_timezone(tz))
    return ts


def to_naive_timestamp(dt: tp.DatetimeLike = "now", **kwargs) -> pd.Timestamp:
    """Parse the datetime as a timezone-naive `pd.Timestamp`."""
    return to_timestamp(dt, **kwargs).tz_localize(None)


def to_datetime(dt: tp.DatetimeLike = "now", **kwargs) -> datetime:
    """Parse the datetime as a `datetime.datetime`.

    Uses `to_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_timestamp(dt, **kwargs).to_pydatetime()


to_local_datetime = partial(to_datetime, tz="tzlocal()")
"""Alias for `to_datetime` with `tz="tzlocal()"`."""

to_utc_datetime = partial(to_datetime, tz="utc")
"""Alias for `to_datetime` with `tz="utc"`."""


def to_tzaware_datetime(dt: tp.DatetimeLike = "now", **kwargs) -> datetime:
    """Parse the datetime as a timezone-aware `datetime.datetime`.

    Uses `to_tzaware_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_tzaware_timestamp(dt, **kwargs).to_pydatetime()


def to_naive_datetime(dt: tp.DatetimeLike = "now", **kwargs) -> datetime:
    """Parse the datetime as a timezone-naive `datetime.datetime`.

    Uses `to_naive_timestamp`."""
    if "unit" not in kwargs:
        kwargs["unit"] = "ms"
    return to_naive_timestamp(dt, **kwargs).to_pydatetime()


def get_min_td_component(td: pd.Timedelta) -> int:
    """Get index of the smallest timedelta component."""
    td_components = td.components
    if td_components.nanoseconds > 0:
        return 6
    if td_components.microseconds > 0:
        return 5
    if td_components.milliseconds > 0:
        return 4
    if td_components.seconds > 0:
        return 3
    if td_components.minutes > 0:
        return 2
    if td_components.hours > 0:
        return 1
    if td_components.days > 0:
        return 0
    return -1


def readable_datetime(
    dt: tp.DatetimeLike = "now",
    drop_tz: tp.Optional[bool] = None,
    freq: tp.Optional[tp.FrequencyLike] = None,
    **kwargs,
) -> str:
    """Get a human-readable datetime string."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]
    readable_cfg = datetime_cfg["readable"]

    if drop_tz is None:
        drop_tz = readable_cfg["drop_tz"]
    if drop_tz:
        ts = to_naive_timestamp(dt, **kwargs)
    else:
        ts = to_timestamp(dt, **kwargs)
    if freq is not None:
        freq = to_freq(freq)
        if isinstance(freq, BaseOffset):
            freq = offset_to_timedelta(freq)
            if freq >= pd.Timedelta(days=1):
                min_freq_component = 0
            else:
                min_freq_component = get_min_td_component(freq)
        else:
            min_freq_component = get_min_td_component(freq)
    else:
        min_freq_component = -1
    td = ts - pd.Timestamp(0, tz=ts.tz)
    ts_components = td.components
    min_ts_component = get_min_td_component(td)
    if min_ts_component == 6 or min_freq_component == 6:
        return ts.strftime(
            "%Y-%m-%d %H:%M:%S.{:03d}{:03d}{:03d}{}".format(
                ts_components.milliseconds,
                ts_components.microseconds,
                ts_components.nanoseconds,
                " %Z" if ts.tz is not None else "",
            )
        )
    if min_ts_component == 5 or min_freq_component == 5:
        return ts.strftime(
            "%Y-%m-%d %H:%M:%S.{:03d}{:03d}{}".format(
                ts_components.milliseconds,
                ts_components.microseconds,
                " %Z" if ts.tz is not None else "",
            )
        )
    if min_ts_component == 4 or min_freq_component == 4:
        return ts.strftime(
            "%Y-%m-%d %H:%M:%S.{:03d}{}".format(
                ts_components.milliseconds,
                " %Z" if ts.tz is not None else "",
            )
        )
    if min_ts_component == 3 or min_freq_component == 3:
        return ts.strftime("%Y-%m-%d %H:%M:%S{}".format(" %Z" if ts.tz is not None else ""))
    if min_ts_component == 2 or min_freq_component == 2:
        return ts.strftime("%Y-%m-%d %H:%M{}".format(" %Z" if ts.tz is not None else ""))
    if min_ts_component == 1 or min_freq_component == 1:
        return ts.strftime("%Y-%m-%d %H:%M{}".format(" %Z" if ts.tz is not None else ""))
    if min_freq_component == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M{}".format(" %Z" if ts.tz is not None else ""))


# ############# Nanoseconds ############# #


def datetime_to_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds."""
    epoch = datetime.fromtimestamp(0, dt.tzinfo)
    return int((dt - epoch).total_seconds() * 1000.0)


def interval_to_ms(interval: str) -> tp.Optional[int]:
    """Convert an interval string to milliseconds."""
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None


def to_ns(obj: tp.ArrayLike, tz_naive_ns: tp.Optional[bool] = None) -> tp.ArrayLike:
    """Convert a datetime, timedelta, integer, or any array-like object to nanoseconds since Unix Epoch."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if tz_naive_ns is None:
        tz_naive_ns = datetime_cfg["tz_naive_ns"]

    if isinstance(obj, time):
        obj = time_to_timedelta(obj)
    if isinstance(obj, pd.Timestamp):
        if tz_naive_ns:
            obj = obj.tz_localize(None).tz_localize("utc")
        obj = obj.to_datetime64()
    if isinstance(obj, BaseOffset):
        obj = pd.Timedelta(obj)
    if isinstance(obj, pd.Timedelta):
        obj = obj.to_timedelta64()
    if isinstance(obj, (datetime, date)):
        obj = np.datetime64(obj)
    if isinstance(obj, timedelta):
        obj = np.timedelta64(obj)
    if isinstance(obj, pd.DatetimeIndex):
        obj = obj.tz_localize(None).tz_localize("utc")
    if isinstance(obj, pd.PeriodIndex):
        obj = obj.to_timestamp()
    if isinstance(obj, pd.DatetimeIndex):
        if tz_naive_ns:
            obj = obj.tz_localize(None).tz_localize("utc")
    if isinstance(obj, pd.Index):
        obj = obj.values

    if not isinstance(obj, np.ndarray):
        new_obj = np.asarray(obj)
    else:
        new_obj = obj
    if np.issubdtype(new_obj.dtype, np.datetime64) and new_obj.dtype != np.dtype("datetime64[ns]"):
        new_obj = new_obj.astype("datetime64[ns]")
    if np.issubdtype(new_obj.dtype, np.timedelta64) and new_obj.dtype != np.dtype("timedelta64[ns]"):
        new_obj = new_obj.astype("timedelta64[ns]")
    new_obj = new_obj.astype(np.int64)
    if new_obj.ndim == 0 and (not isinstance(obj, np.ndarray) or obj.ndim != 0):
        return new_obj.item()
    return new_obj


# ############# Index ############# #


def date_range(
    start: tp.Optional[tp.DatetimeLike] = None,
    end: tp.Optional[tp.DatetimeLike] = None,
    *,
    periods: tp.Optional[int] = None,
    freq: tp.Optional[tp.FrequencyLike] = None,
    tz: tp.Optional[tp.TimezoneLike] = None,
    inclusive: str = "left",
    timestamp_kwargs: tp.KwargsLike = None,
    freq_kwargs: tp.KwargsLike = None,
    timezone_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> pd.DatetimeIndex:
    """Same as `pd.date_range` but preprocesses `start` and `end` with `to_timestamp`,
    `freq` with `to_freq`, and `tz` with `to_timezone`.

    If `start` and `periods` are None, will set `start` to the beginning of the Unix epoch.
    Same if pf `periods` is not None but `start` and `end` are None.

    If `end` and `periods` are None, will set `end` to the current date and time."""
    if timestamp_kwargs is None:
        timestamp_kwargs = {}
    if freq_kwargs is None:
        freq_kwargs = {}
    if timezone_kwargs is None:
        timezone_kwargs = {}
    if freq is None and (start is None or end is None or periods is None):
        freq = "1D"
    if freq is not None:
        freq = to_freq(freq, **freq_kwargs)
    if tz is not None:
        tz = to_timezone(tz, **timezone_kwargs)
    if start is not None:
        start = to_timestamp(start, tz=tz, **timestamp_kwargs)
    if end is not None:
        end = to_timestamp(end, tz=tz, **timestamp_kwargs)
    if periods is None:
        if start is None:
            if tz is not None:
                start = to_timestamp(0, tz=tz, **timestamp_kwargs)
            elif end is not None and end.tz is not None:
                start = to_timestamp(0, tz=end.tz, **timestamp_kwargs)
            else:
                start = to_naive_timestamp(0, **timestamp_kwargs)
        if end is None:
            if tz is not None:
                end = to_timestamp("now", tz=tz, **timestamp_kwargs)
            elif start is not None and start.tz is not None:
                end = to_timestamp("now", tz=start.tz, **timestamp_kwargs)
            else:
                end = to_naive_timestamp("now", **timestamp_kwargs)
    else:
        if start is None and end is None:
            if tz is not None:
                start = to_timestamp(0, tz=tz, **timestamp_kwargs)
            else:
                start = to_naive_timestamp(0, **timestamp_kwargs)
    return pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        inclusive=inclusive,
        **kwargs,
    )


def prepare_dt_index(
    index: tp.IndexLike,
    parse_index: tp.Optional[bool] = None,
    parse_with_dateparser: tp.Optional[bool] = None,
    dateparser_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.Index:
    """Try converting an index to a datetime index.

    If `parse_index` is True and the object has an object data type, will parse with Pandas
    (`parse_index` must be True) and dateparser (in addition `parse_with_dateparser` must be True).

    `dateparser_kwargs` are passed to `dateparser.parse` while `**kwargs` are passed to `pd.to_datetime`.

    For defaults, see `vectorbtpro._settings.datetime`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if parse_index is None:
        parse_index = datetime_cfg["index"]["parse_index"]
    if parse_with_dateparser is None:
        parse_with_dateparser = datetime_cfg["index"]["parse_with_dateparser"]
    dateparser_kwargs = merge_dicts(datetime_cfg["dateparser_kwargs"], dateparser_kwargs)

    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            if parse_index:
                try:
                    parsed_index = pd.to_datetime(index, **kwargs)
                    if not isinstance(parsed_index, pd.Timestamp) and "utc" not in kwargs:
                        parsed_index = pd.to_datetime(index, utc=True, **kwargs)
                    index = [parsed_index]
                except Exception as e:
                    if parse_with_dateparser:
                        try:
                            parsed_index = dateparser.parse(index, **dateparser_kwargs)
                            if parsed_index is None:
                                raise Exception
                            index = pd.to_datetime(parsed_index, **kwargs)
                            index = [index]
                        except Exception as e2:
                            pass
        try:
            index = pd.Index(index)
        except Exception as e:
            index = pd.Index([index])
    if isinstance(index, pd.DatetimeIndex):
        return index
    if index.dtype == object:
        if parse_index:
            try:
                with WarningsFiltered():
                    pd.to_datetime(index[[0]], **kwargs)
                try:
                    parsed_index = pd.to_datetime(index, **kwargs)
                    if (
                        not isinstance(parsed_index, pd.DatetimeIndex)
                        and isinstance(parsed_index[0], datetime)
                        and "utc" not in kwargs
                    ):
                        parsed_index = pd.to_datetime(index, utc=True, **kwargs)
                    return parsed_index
                except Exception as e:
                    if parse_with_dateparser:
                        try:

                            def _parse(x):
                                _parsed_index = dateparser.parse(x, **dateparser_kwargs)
                                if _parsed_index is None:
                                    raise Exception
                                return _parsed_index

                            return pd.to_datetime(index.map(_parse), **kwargs)
                        except Exception as e2:
                            pass
            except Exception as e:
                pass
    return index


def try_align_to_dt_index(source_index: tp.IndexLike, target_index: tp.Index, **kwargs) -> tp.Index:
    """Try aligning an index to another datetime index.

    Keyword arguments are passed to `prepare_dt_index`."""
    source_index = prepare_dt_index(source_index, **kwargs)
    if isinstance(source_index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source_index.tz is None and target_index.tz is not None:
            source_index = source_index.tz_localize(target_index.tz)
        elif source_index.tz is not None and target_index.tz is not None:
            source_index = source_index.tz_convert(target_index.tz)
    return source_index


def try_align_dt_to_index(dt: tp.DatetimeLike, target_index: tp.Index, **kwargs) -> tp.DatetimeLike:
    """Try aligning a datetime-like object to another datetime index.

    Keyword arguments are passed to `to_timestamp`."""
    if not isinstance(target_index, pd.DatetimeIndex):
        return dt
    dt = to_timestamp(dt, **kwargs)
    if dt.tzinfo is None and target_index.tz is not None:
        dt = dt.tz_localize(target_index.tz)
    elif dt.tzinfo is not None and target_index.tz is not None:
        dt = dt.tz_convert(target_index.tz)
    return dt


def auto_detect_freq(index: tp.Index) -> tp.Optional[tp.PandasFrequency]:
    """Auto-detect frequency from a datetime index.

    Returns the minimal frequency if it's being encountered in most of the index."""
    diff_values = index.values[1:] - index.values[:-1]
    if len(diff_values) > 0:
        mini, _, minc = min_count_nb(diff_values)
        if minc / len(index) > 0.5:
            return index[mini + 1] - index[mini]
    return None


def parse_index_freq(index: pd.DatetimeIndex) -> tp.Optional[tp.PandasFrequency]:
    """Parse frequency from a datetime index."""
    if index.freqstr is not None:
        return to_freq(index.freqstr)
    if index.freq is not None:
        return to_freq(index.freq)
    if len(index) >= 3:
        freq = pd.infer_freq(index)
        if freq is not None:
            return to_freq(freq)
    return None


def freq_depends_on_index(freq: tp.FrequencyLike) -> bool:
    """Return whether frequency depends on index."""
    if isinstance(freq, str):
        freq = " ".join(freq.strip().split())
        if freq == "auto":
            return True
        if freq.startswith("index_"):
            return True
    return False


def infer_index_freq(
    index: tp.Index,
    freq: tp.Optional[tp.FrequencyLike] = None,
    allow_offset: bool = True,
    allow_numeric: bool = True,
    freq_from_n: tp.Union[None, bool, int] = None,
) -> tp.Union[None, int, float, tp.PandasFrequency]:
    """Infer frequency of a datetime index if `freq` is None, otherwise convert `freq`.

    If `freq` is "auto", uses `auto_detect_freq`. If `freq` is "index_[method_name]", applies the
    method to the `pd.TimedeltaIndex` resulting from the difference between each pair of index points.
    If `freq_from_n` is a positive or negative number, limits the index to the first or the
    last N index points respectively.

    For defaults, see `vectorbtpro._settings.datetime`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if freq_from_n is None:
        freq_from_n = datetime_cfg["freq_from_n"]
    if isinstance(freq_from_n, bool):
        if freq_from_n:
            raise ValueError("freq_from_n cannot be True")
        freq_from_n = None

    if isinstance(index, pd.DatetimeIndex):
        if freq is None:
            freq = parse_index_freq(index)
        elif isinstance(freq, str):
            if freq_depends_on_index(freq):
                new_freq = parse_index_freq(index)
                if new_freq is not None:
                    freq = new_freq
                else:
                    if freq_from_n is None:
                        index_subset = index
                    else:
                        if freq_from_n >= 0:
                            index_subset = index[:freq_from_n]
                        else:
                            index_subset = index[freq_from_n:]
                    if freq.lower() == "auto":
                        freq = auto_detect_freq(index_subset)
                    else:
                        method_name = freq.lower().replace("index_", "")
                        freq = getattr(index_subset[1:] - index_subset[:-1], method_name)()
    if freq is None:
        return None
    if checks.is_number(freq) and allow_numeric:
        return freq
    return to_freq(freq, allow_offset=allow_offset)


def get_dt_index_gaps(
    index: tp.IndexLike,
    freq: tp.Optional[tp.FrequencyLike] = None,
    skip_index: tp.Optional[tp.IndexLike] = None,
    **kwargs,
) -> tp.Tuple[tp.Index, tp.Index]:
    """Get gaps in a datetime index.

    Returns two indexes: start indexes (inclusive) and end indexes (exclusive).

    Keyword arguments are passed to `prepare_dt_index`."""
    index = prepare_dt_index(index, **kwargs)
    checks.assert_instance_of(index, pd.DatetimeIndex)
    if not index.is_unique:
        raise ValueError("Datetime index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("Datetime index must be monotonically increasing")
    if freq is None:
        freq = infer_index_freq(index, freq="auto", allow_numeric=False, freq_from_n=False)
    else:
        freq = infer_index_freq(index, freq=freq, allow_numeric=False)
    if skip_index is not None:
        skip_index = prepare_dt_index(skip_index, **kwargs)
        checks.assert_instance_of(skip_index, pd.DatetimeIndex)
        skip_bound_start = skip_index.min()
        skip_bound_end = skip_index.max()
        index = index.difference(skip_index)
    else:
        skip_bound_start = None
        skip_bound_end = None
    start_index = index[:-1]
    end_index = index[1:]
    gap_mask = start_index + freq < end_index
    bound_starts = start_index[gap_mask] + freq
    bound_ends = end_index[gap_mask]
    if skip_bound_start is not None and skip_bound_start < index[0]:
        bound_starts = pd.Index([skip_bound_start]).union(bound_starts)
        bound_ends = pd.Index([index[0]]).union(bound_ends)
    if skip_bound_end is not None and skip_bound_end >= index[-1] + freq:
        bound_starts = pd.Index([index[-1] + freq]).union(bound_starts)
        bound_ends = pd.Index([skip_bound_end + freq]).union(bound_ends)
    return bound_starts, bound_ends


def get_rangebreaks(index: tp.IndexLike, **kwargs) -> list:
    """Get `rangebreaks` based on `get_dt_index_gaps`."""
    start_index, end_index = get_dt_index_gaps(index, **kwargs)
    return [dict(bounds=x) for x in zip(start_index, end_index)]
