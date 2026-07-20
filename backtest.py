from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STRATEGY_LABELS = {
    "buy_below": "Acheter au croisement sous la SMA",
    "buy_above": "Acheter au croisement au-dessus de la SMA",
}


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    cost_bps: float = 2.0
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
    required = {"Open", "Close"}
    if not required.issubset(frame.columns):
        raise ValueError("Les données doivent contenir les colonnes Open et Close.")

    frame = frame[["Open", "Close"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna().sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    if frame.empty:
        raise ValueError("Aucune donnée de prix valide n'a été trouvée.")
    return frame


def _state_from_crosses(
    close: pd.Series,
    sma: pd.Series,
    strategy: str,
    active_from: str | pd.Timestamp | None = None,
    cooldown_days: int = 0,
) -> pd.Series:
    above = close > sma
    cross_above = above & ~above.shift(1, fill_value=False)
    cross_below = ~above & above.shift(1, fill_value=False) & sma.notna()

    if strategy == "buy_above":
        entries, exits = cross_above & sma.notna(), cross_below
    elif strategy == "buy_below":
        entries, exits = cross_below, cross_above & sma.notna()
    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    if active_from is not None:
        active = close.index >= pd.Timestamp(active_from)
        entries &= active
        exits &= active

    state = pd.Series(0.0, index=close.index, dtype=float)
    holding = False
    last_entry: pd.Timestamp | None = None

    for date in close.index:
        if holding and bool(exits.at[date]):
            holding = False
        elif not holding and bool(entries.at[date]):
            cooldown_complete = (
                last_entry is None
                or (pd.Timestamp(date) - last_entry).days >= cooldown_days
            )
            if cooldown_complete:
                holding = True
                last_entry = pd.Timestamp(date)
        state.at[date] = float(holding)

    return state


def equity_curve(
    data: pd.DataFrame,
    sma_window: int,
    strategy: str,
    config: BacktestConfig = BacktestConfig(),
    active_from: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Backtest close signals executed at the following session's adjusted open."""
    frame = normalize_ohlc(data)
    frame["SMA"] = frame["Close"].rolling(sma_window, min_periods=sma_window).mean()
    state_at_close = _state_from_crosses(
        frame["Close"],
        frame["SMA"],
        strategy,
        active_from=active_from,
        cooldown_days=config.cooldown_days,
    )

    # A close signal on day t changes the position at open t+1. Open-to-open
    # return ending on t is therefore earned from the state known at close t-2.
    frame["Position"] = state_at_close.shift(1, fill_value=0.0)
    interval_position = state_at_close.shift(2, fill_value=0.0)
    frame["MarketReturn"] = frame["Open"].pct_change().fillna(0.0)
    frame["StrategyReturn"] = interval_position * frame["MarketReturn"]

    transaction = frame["Position"].diff().abs().fillna(frame["Position"].abs())
    frame["StrategyReturn"] -= transaction * (config.cost_bps / 10_000.0)
    frame["Equity"] = config.initial_capital * (1.0 + frame["StrategyReturn"]).cumprod()

    if active_from is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(active_from)].copy()
        if frame.empty:
            raise ValueError("La période demandée ne contient aucune séance.")

    frame["Benchmark"] = config.initial_capital * (
        frame["Close"] / frame["Close"].iloc[0]
    )
    return frame


def _trade_returns(curve: pd.DataFrame, cost_bps: float) -> list[float]:
    changes = curve["Position"].diff().fillna(curve["Position"])
    entry_dates = list(changes.index[changes > 0])
    exit_dates = list(changes.index[changes < 0])
    returns: list[float] = []

    for entry_date in entry_dates:
        later_exits = [date for date in exit_dates if date > entry_date]
        if later_exits:
            exit_date = later_exits[0]
            exit_price = curve.at[exit_date, "Open"]
        else:
            exit_price = curve["Close"].iloc[-1]
        entry_price = curve.at[entry_date, "Open"]
        gross = exit_price / entry_price - 1.0
        returns.append(gross - 2.0 * cost_bps / 10_000.0)
    return returns


def summarize_curve(
    curve: pd.DataFrame,
    sma_window: int,
    strategy: str,
    period_label: str,
    config: BacktestConfig,
) -> dict[str, float | int | str]:
    equity = curve["Equity"]
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / config.initial_capital - 1.0
    cagr = (equity.iloc[-1] / config.initial_capital) ** (1.0 / years) - 1.0
    drawdown = equity / equity.cummax() - 1.0
    trades = _trade_returns(curve, config.cost_bps)
    benchmark_return = curve["Benchmark"].iloc[-1] / config.initial_capital - 1.0
    benchmark_cagr = (1.0 + benchmark_return) ** (1.0 / years) - 1.0

    return {
        "Période": period_label,
        "Stratégie": STRATEGY_LABELS[strategy],
        "Code stratégie": strategy,
        "SMA": sma_window,
        "Rendement total": total_return,
        "Rendement annualisé": cagr,
        "Rendement achat-conservation": benchmark_return,
        "Rendement annualisé achat-conservation": benchmark_cagr,
        "Surperformance": total_return - benchmark_return,
        "Surperformance annualisée": cagr - benchmark_cagr,
        "Drawdown maximal": drawdown.min(),
        "Transactions": len(trades),
        "Taux de réussite": float(np.mean(np.array(trades) > 0)) if trades else np.nan,
        "Exposition": curve["Position"].mean(),
        "Capital final": equity.iloc[-1],
    }


def run_sma_sweep(
    data: pd.DataFrame,
    periods: dict[str, str],
    sma_windows: Iterable[int] = range(150, 251),
    strategies: Iterable[str] = ("buy_below", "buy_above"),
    config: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    frame = normalize_ohlc(data)
    rows: list[dict[str, float | int | str]] = []
    windows = list(sma_windows)

    for period_label, start_date in periods.items():
        sample = frame.loc[frame.index >= pd.Timestamp(start_date)]
        warmup = frame.loc[frame.index < pd.Timestamp(start_date)]
        if len(sample) <= 2 or len(warmup) < max(windows):
            continue
        for strategy in strategies:
            for window in windows:
                curve = equity_curve(
                    frame, window, strategy, config, active_from=start_date
                )
                rows.append(summarize_curve(curve, window, strategy, period_label, config))

    if not rows:
        raise ValueError("Il n'y a pas assez de données pour les périodes et SMA choisies.")
    return pd.DataFrame(rows)
