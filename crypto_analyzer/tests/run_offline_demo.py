#!/usr/bin/env python3
"""
tests/run_offline_demo.py
Runs the whole pipeline (indicators + pattern detection + S/R + stop loss +
backtest) on synthetic data — no internet needed. For a quick test after any
code change, or to see what the report looks like, run this:

    python tests/run_offline_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.sample_data import generate_synthetic_candles
from analysis.analyzer import build_report
from main import print_report


def run():
    print("در حال تولید داده‌ی مصنوعی و اجرای کل پایپ‌لاین (آفلاین، بدون اینترنت)...")
    df = generate_synthetic_candles(n=300, start_price=50000.0, seed=42)
    report = build_report(df, symbol_label="TEST/USD (داده‌ی مصنوعی)", timeframe="1d")
    print_report(report)
    print("✅ پایپ‌لاین بدون خطا اجرا شد.\n")
    return report


if __name__ == "__main__":
    run()
