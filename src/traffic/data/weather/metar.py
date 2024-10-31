import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Optional, Tuple, Union

from metar import Metar

import numpy as np
import pandas as pd

from ...core.mixins import DataFrameMixin
from ...core.structure import Airport
from ...core.time import timelike, to_datetime
from .. import client

list_ap = Union[List[str], List[Airport]]

_log = logging.getLogger(__name__)


class METAR:  # coverage: ignore
    def __init__(self, airport: Union[str, Airport, list_ap]):
        if isinstance(airport, str):
            self.airports = [airport]
        elif isinstance(airport, Airport):
            self.airports = [airport.icao]
        else:
            self.airports = []
            for ap in airport:
                if isinstance(ap, str):
                    self.airports.append(ap)
                else:
                    self.airports.append(ap.icao)

    def get(
        self, start: Optional[timelike] = None, stop: Optional[timelike] = None
    ) -> pd.DataFrame:
        """Retrieve METAR infos.

        Parameters
        ----------
        start : Optional[timelike]
            Time for which METAR info are retrieved. If no hour is specified,
            data are retrieved for the day. If no time is specified, the
            current METAR information are retrieved.
        stop : Optional[timelike]
            Time until which METAR info are retrieved.
            If no stop time is specified, data are retrieved for 24 hours.

        Returns
        -------
        pd.DataFrame
            DataFrame containing METAR information.
        """
        start = (
            to_datetime(start)
            if start is not None
            else datetime.now(tz=timezone.utc)
        )
        stop = (
            to_datetime(stop)
            if stop is not None
            else start + timedelta(hours=24)
        )
        if stop == start:
            stop += timedelta(hours=1)

        str_ap = ""
        for ap in self.airports:
            str_ap += f"station={ap}&"
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        url += str_ap
        url += (
            f"data=metar&year1={start.year}&"
            f"month1={start.month}&"
            f"day1={start.day}&"
            f"hour1={start.hour}&"
            f"year2={stop.year}&"
            f"month2={stop.month}&"
            f"day2={stop.day}&"
            f"hour2={stop.hour}&"
            f"tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&"
            f"missing=M&trace=T&direct=no&report_type=3&report_type=4"
        )
        c = client.get(url)
        c.raise_for_status()
        list_ = c.content.decode("utf-8").strip().split("\n")
        df_metar = pd.DataFrame.from_records(
            [
                vars(
                    Metar.Metar(
                        m.split(",")[-1],
                        month=datetime.strptime(
                            m.split(",")[1], "%Y-%m-%d %H:%M"
                        )
                        .replace(tzinfo=timezone.utc)
                        .month,
                        year=datetime.strptime(
                            m.split(",")[1], "%Y-%m-%d %H:%M"
                        )
                        .replace(tzinfo=timezone.utc)
                        .year,
                        strict=False,
                        utcdelta=timedelta(hours=0),
                    )
                )
                for m in list_[1:]
            ]
        )
        df_metar["time"] = df_metar["time"].dt.tz_localize("utc")
        df_metar["airport"] = (
            df_metar["code"].str.split(" ").apply(lambda x: x[0])
        )
        return df_metar.drop(
            columns=[c for c in df_metar.columns if c.startswith("_")]
        )


class MetarsIem(DataFrameMixin):
    """
    A wrapper around a pandas DataFrame providing basic metar functionality.
    Data can be loaded from a file on disk or directly downloaded from IEM
    (see https://mesonet.agron.iastate.edu/request/download.phtml)

    The minimum set of required features are:

    - ``station``: the identifier of the station which reported the metar;
    - ``valid``: a timestamp of the moment the metar was reported;
    - ``elevation``: the altitude MSL of the station;
    - ``wind_direction``: as reported in metar - ]0, 360] in tens intervals;
    - ``wind_speed``: as reported in metar, in knots;
    - ``temperature``: as reported in metar, in Celsius;
    - ``sea_level_pressure``: as reported in metar, assumed to be in hectopascal

    """

    cache_dir: Path

    _metar_columns: ClassVar[List[str]] = [
        "station",
        "valid",
        "elevation",
        "wind_direction",
        "wind_speed",
        "temperature",
        "sea_level_pressure",
    ]

    def __init__(self, data: pd.DataFrame | None = None):
        if data is None:
            return

        super().__init__(data)

    @classmethod
    def from_file(
        cls, filename: Path | str, **kwargs: Any
    ) -> "MetarsIem | None":
        """
        Loads metar data from a file on disk.

        :param filename: A valid file path.
        :param kwargs: Arguments to be forwarded to the pd.read_*** functions.

        :return: A Metars instance with the loaded metars or None
        """

        path = Path(filename)

        tentative = super().from_file(filename, **kwargs)

        if tentative is not None:
            for col in cls._metar_columns:
                if col not in tentative.data.columns:
                    _log.warning(
                        f"Column {col} is mandatory for METAR data "
                        f"but was not found in {path.as_posix()}"
                    )
                    return None
            tentative["valid"] = pd.to_datetime(tentative["valid"])  # type: ignore
            return tentative

        _log.warning(f"{path.suffixes} extension is not supported")
        return None

    @classmethod
    def get(
        cls,
        stations: str | Iterable[str],
        start_time: timelike | None = None,
        stop_time: timelike | None = None,
    ) -> "MetarsIem | None":
        """
        Download metar data from the IEM website.

        :param stations: The identifiers of the stations for which to
        download data.
        :param start_time, stop_time: The time interval of interest.

        - If no times are passed, the last two hours will be downloaded

        - If only start_time is passed, stop_time is set to start_time + 2 hours

        - If only stop_time is passed, start_time is set to stop_time - 2 hours
        """

        start_time, stop_time = cls._start_stop_time(start_time, stop_time)

        if stop_time <= start_time:  # type: ignore
            return None

        if isinstance(stations, str):
            stations = [stations]

        cumul: List[pd.DataFrame] = []
        for station in stations:
            if path := cls._get_cache_file(station, start_time, stop_time):
                cumul.append(
                    pd.read_parquet(path).query(
                        "@start_time <= valid <= @stop_time"
                    )
                )
            else:
                cumul.append(cls._iem_download(station, start_time, stop_time))

        if not cumul:
            return None

        return MetarsIem(pd.concat(cumul))

    def __getitem__(self, airport: str) -> "MetarsIem | None":
        df = self.data.loc[self.data["station"] == airport]

        if df.shape[0] == 0:
            return None
        return self.__class__(df)

    def fetch(
        self,
        stations: str | Iterable[str],
        start_time: timelike | None = None,
        stop_time: timelike | None = None,
        enable_download: bool = True,
    ) -> "MetarsIem | None":
        """
        This method is similar to the class method 'get', but it checks if the
        requested data is already available in the class instance, returning
        that subset of data instead if present.

        :param enable_download: if set to False, data will never be
         downloaded and only subset of the existing data will be returned,
         if present.

        For information on the other parameters check the class method 'get'.
        """

        start_time, stop_time = self._start_stop_time(start_time, stop_time)

        if stop_time <= start_time:  # type: ignore
            return None

        if isinstance(stations, str):
            stations = [stations]

        has_data = hasattr(self, "data")
        if not has_data:
            if not enable_download:
                return None

            new_metar = MetarsIem.get(stations, start_time, stop_time)

            if new_metar is not None:
                self.data = new_metar.data
                return self

            return None

        df = self.data.loc[self.data["station"].isin(stations)].query(
            "@start_time <= valid <= @stop_time"
        )

        if df.empty:
            new_metar = self.get(stations, start_time, stop_time)
            if new_metar is not None:
                self.data = pd.concat([self.data, new_metar.data])
                return self.__class__(new_metar.data)

            return None

        return self.__class__(
            df.drop_duplicates(subset=["valid", "station"])
        )  # TEMP drop duplicates, fix for strange behavior

    def compute_wind(self) -> "MetarsIem":
        """
        Enrich the underlying dataframe with wind_u and wind_v,
        calculated based on wind_direction and wind_speed.
        """

        return self.assign(
            wind_rad=lambda df: np.radians(270 - df["wind_direction"]),
            wind_u=lambda df: np.cos(df["wind_rad"]) * df["wind_speed"],
            wind_v=lambda df: np.sin(df["wind_rad"]) * df["wind_speed"],
        ).drop(columns="wind_rad")

    @classmethod
    def _get_cache_file(
        cls, station: str, start_time: pd.Timestamp, stop_time: pd.Timestamp
    ) -> Path | None:
        """
        Searches the cache directory for
        "metar_iem_<station>_<start_time>_<end_time>.parquet" files
        Returns the first file with station
        and <start_time> <= start_time and <end_time> >= end_time
        """

        # Iterate all entries in cache directory
        for file in (cls.cache_dir / "metars").iterdir():
            # Check if entry is a metar_iem parquet file
            if (
                file.is_file()
                and file.suffix == ".parquet"
                and "metar_iem" in file.stem
            ):
                strs = file.stem.split("_")  # split name at _
                if len(strs) < 5:
                    continue
                idx = strs.index("iem")
                if idx >= len(strs) - 3:  # not enough values after iem
                    continue

                if strs[idx + 1] != station:  # file for another station
                    continue
                try:
                    if (
                        pd.to_datetime(
                            strs[idx + 2], format="%Y%m%d%H", utc=True
                        )
                        <= start_time
                        and pd.to_datetime(
                            strs[idx + 3], format="%Y%m%d%H", utc=True
                        )
                        >= stop_time
                    ):
                        return file
                except ValueError:
                    continue

        return None

    @classmethod
    def _iem_download(
        cls, station: str, start_time: pd.Timestamp, stop_time: pd.Timestamp
    ) -> pd.DataFrame:
        from .. import client

        start_time = to_datetime(start_time)
        stop_time = to_datetime(stop_time)

        # Downloading between the same timestamp returns data for 2 days
        # Prevent this by separating timestamps by at least 1 minute
        if start_time.round("1min") == stop_time.round("1min"):
            stop_time += timedelta(minutes=1)

        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        payload = {
            "data": ["drct", "sknt", "tmpc", "dwpc", "relh", "mslp", "alti"],
            "station": station,
            "year1": start_time.year,
            "year2": stop_time.year,
            "month1": start_time.month,
            "month2": stop_time.month,
            "day1": start_time.day,
            "day2": stop_time.day,
            "hour1": start_time.hour,
            "hour2": stop_time.hour,
            "minute1": start_time.minute,
            "minute2": stop_time.minute,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "elev": "yes",
            "missing": "M",
            "trace": "T",
            "direct": "no",
            "report_type": [3, 4],
        }

        time.sleep(0.5)
        res = client.get(url, params=payload)
        res.raise_for_status()
        cols = [
            "station",
            "valid",
            "elevation",
            "wind_direction",
            "wind_speed",
            "temperature",
            "dewpoint",
            "relative_humidity",
            "sea_level_pressure",
            "press_temp_1",
        ]

        df = pd.DataFrame(
            [
                row.split(",")
                for row in res.content.decode("utf-8").split("\n")[1:-1]
            ],
            columns=cols,
        )
        if df.empty:
            df = df.drop(columns="press_temp_1")
        else:
            cols_floats = [
                "elevation",
                "wind_direction",
                "wind_speed",
                "temperature",
                "dewpoint",
                "relative_humidity",
                "sea_level_pressure",
                "press_temp_1",
            ]
            df[cols_floats] = df[cols_floats].apply(
                pd.to_numeric, errors="coerce"
            )
            df["valid"] = pd.to_datetime(df["valid"], utc=True)
            df["press_temp_2"] = df.apply(
                lambda r: r.sea_level_pressure
                if ~np.isnan(r.sea_level_pressure)
                else round(r.press_temp_1 * 33.86),
                axis=1,
            )
            df["sea_level_pressure"] = df["press_temp_2"]
            df = df.drop(columns=["press_temp_1", "press_temp_2"])
            df = df.fillna(
                {
                    "wind_direction": 0.0,
                    "wind_speed": 0.0,
                }
            )
            df.to_parquet(
                cls.cache_dir / "metars" / f"metar_iem_{station}_"
                f"{start_time.strftime('%Y%m%d%H')}_"
                f"{stop_time.strftime('%Y%m%d%H')}.parquet"
            )
        return df

    @staticmethod
    def _start_stop_time(
        start_time: timelike | None = None,
        stop_time: timelike | None = None,
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if not stop_time:
            # Defaults to start_time + 2 hours or now rounded to the hour after
            stop_time = (
                to_datetime(start_time) + pd.Timedelta(hours=2)
                if start_time
                else pd.Timestamp.now(tz="utc").ceil("1hour")
            )
        else:
            stop_time = to_datetime(stop_time)

        if not start_time:
            start_time = to_datetime(stop_time) - pd.Timedelta(hours=2)
        else:
            start_time = to_datetime(start_time)

        return start_time.floor("1h"), stop_time.ceil("1h")  # type: ignore
