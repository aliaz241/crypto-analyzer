#!/usr/bin/env python3
"""
tests/run_offline_demo.py
کل پایپ‌لاین (اندیکاتور + الگویابی + S/R + حد ضرر + بک‌تست) را روی داده‌ی
مصنوعی اجرا می‌کند — بدون نیاز به اینترنت. برای تست سریع بعد از هر تغییر در کد
یا برای دیدن ظاهر گزارش، همین را اجرا کن:

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
