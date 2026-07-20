from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "cross_below": "Croisement sous la SMA",
    "cross_above": "Croisement au-dessus de la SMA",
}

HORIZONS = (
    ("10 jours", "sessions", 10),
    ("25 jours", "sessions", 25),
    ("50 jours", "sessions", 50),
    ("100 jours", "sessions", 100),
    ("200 jours", "sessions", 200),
    ("1 an", "years", 1),
    ("2 ans", "years", 2),
    ("3 ans", "years", 3),
    ("5 ans", "years", 5),
    ("10 ans", "years", 10),
)
HORIZON_ORDER = [label for label, _, _ in HORIZONS]


@dataclass(frozen=True)
class StudyConfig:
    cooldown_days: int = 30


def normalize_ohlc(data: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Return a clean adjusted OHLC frame, including yfinance MultiIndex output."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker and ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)

    frame.columns = [str(col).title() for col in frame.columns]
    if not {"Open", "Close"}.issubset(frame.columns):
        raise ValueError("Les données doivent contenir les colonnes Open et Close.")

    frame = frame[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna().sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    if frame.empty:
        raise ValueError("Aucune donnée de prix valide n'a été trouvée.")
    return frame


def _candidate_signals(
    close: pd.Series,
    sma: pd.Series,
    strategy: str,
) -> pd.Series:
    above = close > sma
    previous_above = above.shift(1, fill_value=False)
    if strategy == "cross_above":
        return above & ~previous_above & sma.notna()
    if strategy == "cross_below":
        return ~above & previous_above & sma.notna()
    raise ValueError(f"Stratégie inconnue : {strategy}")


def signal_dates(
    data: pd.DataFrame,
    sma_window: int,
    strategy: str,
    active_from: str | pd.Timestamp,
    cooldown_days: int = 30,
) -> pd.DatetimeIndex:
    """Find close signals and suppress later signals during the calendar cooldown."""
    frame = normalize_ohlc(data)
    sma = frame["Close"].rolling(sma_window, min_periods=sma_window).mean()
    candidates = _candidate_signals(frame["Close"], sma, strategy)
    candidates &= frame.index >= pd.Timestamp(active_from)

    accepted: list[pd.Timestamp] = []
    last_signal: pd.Timestamp | None = None
    for date in frame.index[candidates]:
        date = pd.Timestamp(date)
        if last_signal is None or (date - last_signal).days >= cooldown_days:
            accepted.append(date)
            last_signal = date
    return pd.DatetimeIndex(accepted)


def _target_location(
    index: pd.DatetimeIndex,
    entry_location: int,
    entry_date: pd.Timestamp,
    horizon_type: str,
    horizon_value: int,
) -> int | None:
    if horizon_type == "sessions":
        location = entry_location + horizon_value
        return location if location < len(index) else None

    target_date = entry_date + pd.DateOffset(years=horizon_value)
    location = int(index.searchsorted(target_date, side="left"))
    return location if location < len(index) else None


def event_returns(
    data: pd.DataFrame,
    sma_window: int,
    strategy: str,
    active_from: str | pd.Timestamp,
    period_label: str,
    config: StudyConfig = StudyConfig(),
) -> pd.DataFrame:
    """Measure adjusted returns after each accepted signal; no exit rule is used."""
    frame = normalize_ohlc(data)
    signals = signal_dates(
        frame, sma_window, strategy, active_from, config.cooldown_days
    )
    rows: list[dict[str, object]] = []

    for signal_date in signals:
        signal_location = int(frame.index.get_loc(signal_date))
        entry_location = signal_location + 1
        if entry_location >= len(frame):
            continue
        entry_date = pd.Timestamp(frame.index[entry_location])
        entry_price = float(frame["Open"].iloc[entry_location])

        for horizon_label, horizon_type, horizon_value in HORIZONS:
            target_location = _target_location(
                frame.index,
                entry_location,
                entry_date,
                horizon_type,
                horizon_value,
            )
            if target_location is None:
                continue
            observation_date = pd.Timestamp(frame.index[target_location])
            observation_price = float(frame["Close"].iloc[target_location])
            rows.append(
                {
                    "Période": period_label,
                    "Stratégie": STRATEGY_LABELS[strategy],
                    "Code stratégie": strategy,
                    "SMA": sma_window,
                    "Date du signal": signal_date,
                    "Date d'entrée": entry_date,
                    "Prix d'entrée": entry_price,
                    "Horizon": horizon_label,
                    "Date d'observation": observation_date,
                    "Prix d'observation": observation_price,
                    "Rendement": observation_price / entry_price - 1.0,
                }
            )
    return pd.DataFrame(rows)


def run_event_study(
    data: pd.DataFrame,
    periods: dict[str, str],
    sma_windows: Iterable[int] = range(150, 251),
    strategies: Iterable[str] = ("cross_below", "cross_above"),
    config: StudyConfig = StudyConfig(),
) -> pd.DataFrame:
    frame = normalize_ohlc(data)
    windows = list(sma_windows)
    all_events: list[pd.DataFrame] = []

    for period_label, start_date in periods.items():
        warmup = frame.loc[frame.index < pd.Timestamp(start_date)]
        if len(warmup) < max(windows):
            continue
        for strategy in strategies:
            for window in windows:
                events = event_returns(
                    frame,
                    window,
                    strategy,
                    start_date,
                    period_label,
                    config,
                )
                if not events.empty:
                    all_events.append(events)

    if not all_events:
        raise ValueError("Aucun signal exploitable n'a été trouvé.")
    result = pd.concat(all_events, ignore_index=True)
    result["Horizon"] = pd.Categorical(
        result["Horizon"], categories=HORIZON_ORDER, ordered=True
    )
    return result


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    summary = (
        events.groupby(
            ["Période", "Stratégie", "Code stratégie", "SMA", "Horizon"],
            observed=True,
        )["Rendement"]
        .agg(
            Observations="count",
            **{
                "Rendement moyen": "mean",
                "Rendement médian": "median",
                "Meilleur rendement": "max",
                "Pire rendement": "min",
                "Écart-type": "std",
            },
        )
        .reset_index()
    )
    win_rate = (
        events.assign(Gagnant=events["Rendement"] > 0)
        .groupby(
            ["Période", "Stratégie", "Code stratégie", "SMA", "Horizon"],
            observed=True,
        )["Gagnant"]
        .mean()
        .rename("Taux positif")
        .reset_index()
    )
    return summary.merge(
        win_rate,
        on=["Période", "Stratégie", "Code stratégie", "SMA", "Horizon"],
    )

