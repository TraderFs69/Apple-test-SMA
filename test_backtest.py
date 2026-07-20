import unittest

import numpy as np
import pandas as pd

from backtest import BacktestConfig, equity_curve, normalize_ohlc, run_sma_sweep


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range("2018-01-01", periods=1_400)
        trend = np.linspace(50, 130, len(dates))
        wave = 14 * np.sin(np.arange(len(dates)) / 25)
        close = trend + wave
        open_ = close * (1 + 0.001 * np.cos(np.arange(len(dates)) / 7))
        cls.data = pd.DataFrame({"Open": open_, "Close": close}, index=dates)

    def test_normalize_requires_prices(self):
        with self.assertRaises(ValueError):
            normalize_ohlc(pd.DataFrame({"Volume": [1, 2]}))

    def test_curve_has_no_lookahead_entry(self):
        curve = equity_curve(self.data, 150, "buy_above")
        self.assertEqual(curve["Position"].iloc[:150].sum(), 0)
        self.assertTrue((curve["Equity"] > 0).all())

    def test_full_sweep_shape_and_metrics(self):
        results = run_sma_sweep(
            self.data,
            {"Test": "2019-01-01"},
            range(150, 153),
            config=BacktestConfig(10_000, 2),
        )
        self.assertEqual(len(results), 6)
        self.assertEqual(set(results["SMA"]), {150, 151, 152})
        self.assertTrue(results["Drawdown maximal"].le(0).all())

    def test_period_uses_prior_prices_as_sma_warmup(self):
        curve = equity_curve(
            self.data, 150, "buy_above", active_from="2019-01-01"
        )
        self.assertGreaterEqual(curve.index.min(), pd.Timestamp("2019-01-01"))
        self.assertFalse(curve["SMA"].iloc[0:5].isna().any())

    def test_entry_signals_respect_cooldown(self):
        curve = equity_curve(
            self.data,
            150,
            "buy_above",
            BacktestConfig(cooldown_days=90),
        )
        entries = curve.index[curve["Position"].diff().fillna(0) > 0]
        if len(entries) > 1:
            gaps = pd.Series(entries[1:] - entries[:-1]).dt.days
            self.assertTrue(gaps.ge(90).all())


if __name__ == "__main__":
    unittest.main()
