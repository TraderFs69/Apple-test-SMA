import unittest

import numpy as np
import pandas as pd

from backtest import (
    HORIZON_ORDER,
    StudyConfig,
    event_returns,
    normalize_ohlc,
    run_event_study,
    signal_dates,
    summarize_events,
)


class EventStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range("2008-01-01", periods=4_600)
        trend = np.linspace(50, 180, len(dates))
        wave = 18 * np.sin(np.arange(len(dates)) / 24)
        close = trend + wave
        open_ = close * (1 + 0.001 * np.cos(np.arange(len(dates)) / 7))
        cls.data = pd.DataFrame({"Open": open_, "Close": close}, index=dates)

    def test_normalize_requires_prices(self):
        with self.assertRaises(ValueError):
            normalize_ohlc(pd.DataFrame({"Volume": [1, 2]}))

    def test_signal_cooldown(self):
        signals = signal_dates(
            self.data, 150, "cross_above", "2010-01-01", cooldown_days=90
        )
        if len(signals) > 1:
            gaps = pd.Series(signals[1:] - signals[:-1]).dt.days
            self.assertTrue(gaps.ge(90).all())

    def test_entry_is_next_session(self):
        events = event_returns(
            self.data,
            150,
            "cross_below",
            "2010-01-01",
            "Test",
            StudyConfig(30),
        )
        first = events.iloc[0]
        signal_location = self.data.index.get_loc(first["Date du signal"])
        self.assertEqual(first["Date d'entrée"], self.data.index[signal_location + 1])

    def test_long_horizons_only_when_available(self):
        events = event_returns(
            self.data, 150, "cross_above", "2024-01-01", "Test"
        )
        if not events.empty:
            late = events[events["Date d'entrée"] > self.data.index[-1] - pd.Timedelta(days=365)]
            self.assertFalse((late["Horizon"].astype(str) == "1 an").any())

    def test_study_and_summary(self):
        events = run_event_study(
            self.data,
            {"Depuis 2010": "2010-01-01"},
            range(150, 153),
            config=StudyConfig(30),
        )
        summary = summarize_events(events)
        self.assertFalse(events.empty)
        self.assertFalse(summary.empty)
        self.assertTrue(set(events["Horizon"].astype(str)).issubset(HORIZON_ORDER))
        self.assertTrue(events["Rendement"].notna().all())


if __name__ == "__main__":
    unittest.main()

