#!/usr/bin/env python3
"""
main.py
ابزار خط‌فرمان تحلیل تکنیکال ارز دیجیتال (نسخه‌ی ۲).

استفاده:
    python main.py                       (منوی شماره‌دار فارسی)
    python main.py --symbol btc --timeframe 4h
    python main.py --symbol btc --timeframe 1d --chart   (همراه تصویر نمودار)
    python main.py --scan                (اسکن بازار و پیشنهاد امروز)
    python main.py --symbol bitcoin --vs usd --timeframe 4h --source coingecko
"""

import argparse
import os
import sys
from datetime import datetime

from analysis.analyzer import (AnalysisError, build_report, fetch_candles,
                               fetch_candles_binance)
from data.coingecko_client import CoinGeckoError
from data.binance_client import BinanceError
import config


SEPARATOR = "─" * 50


def _fmt(value, decimals=4):
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


def print_report(report: dict):
    print("\n" + "=" * 50)
    print(f"  تحلیل تکنیکال: {report['symbol'].upper()}  |  تایم‌فریم: {report['timeframe']}")
    print(f"  تعداد کندل تحلیل‌شده: {report['candle_count']}  |  آخرین آپدیت: {report['last_updated']}")
    print("=" * 50)

    ind = report["indicators"]
    print(f"\n💰 قیمت فعلی: {_fmt(ind['price'])}")

    print(f"\n{SEPARATOR}\n📊 اندیکاتورها\n{SEPARATOR}")
    rsi_note = ""
    if ind["rsi"] is not None:
        if ind["rsi"] >= 70:
            rsi_note = " (اشباع خرید)"
        elif ind["rsi"] <= 30:
            rsi_note = " (اشباع فروش)"
    print(f"RSI(14): {_fmt(ind['rsi'], 1)}{rsi_note}")

    macd = ind["macd"]
    macd_bias = "—"
    if macd["histogram"] is not None:
        macd_bias = "صعودی" if macd["histogram"] > 0 else "نزولی"
    print(f"MACD: {_fmt(macd['macd'], 6)}  |  Signal: {_fmt(macd['signal'], 6)}  |  Histogram: {_fmt(macd['histogram'], 6)} ({macd_bias})")

    stoch = ind["stochastic"]
    print(f"Stochastic: K={_fmt(stoch['k'], 1)}  D={_fmt(stoch['d'], 1)}")

    cci_note = ""
    if ind["cci"] is not None:
        if ind["cci"] >= 100:
            cci_note = " (اشباع خرید)"
        elif ind["cci"] <= -100:
            cci_note = " (اشباع فروش)"
    print(f"CCI: {_fmt(ind['cci'], 1)}{cci_note}")

    wr_note = ""
    if ind["williams_r"] is not None:
        if ind["williams_r"] >= -20:
            wr_note = " (اشباع خرید)"
        elif ind["williams_r"] <= -80:
            wr_note = " (اشباع فروش)"
    print(f"Williams %R: {_fmt(ind['williams_r'], 1)}{wr_note}")

    bb = ind["bollinger"]
    print(f"Bollinger Bands: پایین={_fmt(bb['lower'])}  میانه={_fmt(bb['middle'])}  بالا={_fmt(bb['upper'])}")

    kc = ind["keltner"]
    print(f"Keltner Channel: پایین={_fmt(kc['lower'])}  میانه={_fmt(kc['middle'])}  بالا={_fmt(kc['upper'])}")

    adx_note = ""
    if ind["adx"] is not None:
        adx_note = " (روند قوی)" if ind["adx"] >= 25 else " (بدون روند مشخص/رنج)"
    print(f"ADX: {_fmt(ind['adx'], 1)}{adx_note}")
    print(f"ATR: {_fmt(ind['atr'], 6)}")

    if ind["obv_trend"] is not None:
        obv_fa = "صعودی" if ind["obv_trend"] == "up" else "نزولی"
        print(f"روند حجم (OBV, 5 کندل اخیر): {obv_fa}")

    print(f"\n{SEPARATOR}\n📈 سطوح حمایت و مقاومت\n{SEPARATOR}")
    sr = report["support_resistance"]
    if sr["resistance"]:
        print("مقاومت (نزدیک‌ترین به دورترین):")
        for r in sr["resistance"][:4]:
            print(f"  • {_fmt(r['level'])}  (قدرت سطح: {r['touches']} برخورد)")
    else:
        print("مقاومت معناداری در داده‌ی فعلی یافت نشد.")

    if sr["support"]:
        print("حمایت (نزدیک‌ترین به دورترین):")
        for s in sr["support"][:4]:
            print(f"  • {_fmt(s['level'])}  (قدرت سطح: {s['touches']} برخورد)")
    else:
        print("حمایت معناداری در داده‌ی فعلی یافت نشد.")

    print(f"\n{SEPARATOR}\n🕯️ الگوهای کندلی اخیر\n{SEPARATOR}")
    candle_patterns = report["latest_candlestick_patterns"]
    if candle_patterns:
        backtest_c = report["backtest"]["candlestick"]
        for p in candle_patterns:
            bias_fa = {"bullish": "صعودی", "bearish": "نزولی", "neutral": "خنثی"}[p["bias"]]
            bt = backtest_c.get(p["name"])
            bt_note = ""
            if bt:
                reliability = "" if bt["reliable_sample"] else "  ⚠️ نمونه کم، آماری قابل اتکا نیست"
                bt_note = (f"  →  نرخ موفقیت تاریخی: {bt['success_rate_pct']}% "
                           f"({bt['wins']} از {bt['total_occurrences']} رخداد مشابه){reliability}")
            print(f"  • {p['fa']} ({bias_fa}) در کندل {p['index']}{bt_note}")
    else:
        print("الگوی کندلی خاصی در کندل‌های اخیر شناسایی نشد.")

    print(f"\n{SEPARATOR}\n📐 الگوهای نموداری اخیر\n{SEPARATOR}")
    chart_patterns_found = report["latest_chart_patterns"]
    if chart_patterns_found:
        backtest_chart = report["backtest"]["chart"]
        for p in chart_patterns_found:
            bias_fa = {"bullish": "صعودی", "bearish": "نزولی"}.get(p["bias"], p["bias"])
            bt = backtest_chart.get(p["name"])
            bt_note = ""
            if bt:
                reliability = "" if bt["reliable_sample"] else "  ⚠️ نمونه کم، آماری قابل اتکا نیست"
                bt_note = (f"\n      نرخ موفقیت تاریخی این الگو: {bt['success_rate_pct']}% "
                           f"({bt['wins']} از {bt['total_occurrences']} رخداد مشابه){reliability}")
            print(f"  • {p['fa']} ({bias_fa})")
            print(f"      نقطه‌ی تکمیل: {p['completion_index']}")
            print(f"      خط گردن: {_fmt(p['neckline'])}  |  هدف: {_fmt(p['target'])}  |  بطلان: {_fmt(p['invalidation'])}{bt_note}")
    else:
        print("الگوی نموداری خاصی در کندل‌های اخیر تکمیل نشده است.")

    print(f"\n{SEPARATOR}\n🛑 پیشنهاد حد ضرر\n{SEPARATOR}")
    for label, data in [("اگر لانگ (خرید) باشی", report["stop_loss_long"]),
                         ("اگر شورت (فروش) باشی", report["stop_loss_short"])]:
        rec = data.get("recommended")
        if rec and rec.get("stop_loss") is not None:
            print(f"{label}: {_fmt(rec['stop_loss'])}  "
                  f"(روش: {rec['method']}, فاصله: {rec['distance_pct']:.2f}%)")
        else:
            print(f"{label}: قابل محاسبه نبود ({data.get('reason', '—')})")

    print(f"\n{SEPARATOR}")
    print("⚠️  نرخ‌های موفقیت بالا آمار تاریخیِ عملکرد همین الگو روی همین دارایی/تایم‌فریم")
    print("    هستند، نه پیش‌بینی قطعی آینده. همیشه با مدیریت ریسک معامله کن.")
    print(SEPARATOR + "\n")



def _fmt_price(value, decimals=4):
    """مثل _fmt ولی برای قیمت‌های خیلی کوچک (مثل PEPE) رقم معنادار نشان می‌دهد."""
    if value is None:
        return "—"
    if 0 < abs(value) < 0.01:
        return f"{value:,.8f}"
    return f"{value:,.{decimals}f}"


def _ev_part(c) -> str:
    """بخش EV خط رتبه‌بندی اسکن؛ برای ورودی کشِ بدون معامله (expectancy
    خالی، مثل ترکیب‌های تازه‌ی فارکس) قالب‌بندی :+ روی None می‌ترکید —
    این‌جا امن شده و فقط وقتی EV عددی هست بخش اضافه می‌شود."""
    ev = c.get("ev")
    if not ev or ev.get("expectancy_r") is None:
        return ""
    return f" | EV: {ev['expectancy_r']:+}R (n={ev['n_trades']})"


def _analyze(symbol, vs, timeframe, source):
    """داده را از منبع انتخابی می‌گیرد و (df, report) برمی‌گرداند."""
    if source == "binance":
        df = fetch_candles_binance(symbol, timeframe)
    elif source in ("forex", "market"):
        from analysis.analyzer import fetch_market
        df = fetch_market(symbol, timeframe,
                          market="forex" if source == "forex" else None)
    else:
        df = fetch_candles(symbol, vs, timeframe)
    return df, build_report(df, symbol, timeframe)


def _render_and_announce(df, report):
    """نمودار را رسم و مسیر فایل ذخیره‌شده را چاپ می‌کند."""
    import charting
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report['symbol'].upper()}_{report['timeframe']}_{ts}.png"
    path = charting.render_chart(
        df, report, os.path.join(config.CHART_OUTPUT_DIR, filename))
    print(f"\n🖼️ تصویر نمودار ذخیره شد: {path}")
    return path


def _print_llm_block(c, indent=""):
    """بلوک نمایشی افزودنی بازبینی LLM برای یک نامزد (فقط اگر کلید llm داشته باشد)."""
    llm = c.get("llm")
    if not llm:
        return
    votes = llm.get("llm_votes") or {}
    votes_txt = "، ".join(f"{k}: {v}" for k, v in votes.items()) or "—"
    conf = llm.get("llm_confidence")
    conf_txt = f"{conf}٪" if conf is not None else "—"
    print(f"{indent}🧠 بازبینی LLM: تصمیم {llm.get('llm_decision')} | "
          f"اطمینان {conf_txt} | آرا: {votes_txt}")
    if llm.get("llm_crosscheck_disagreement"):
        print(f"{indent}⚠️ کراس‌چک مستقل مخالف بود (یک پله تنزل اعمال شد).")
    if llm.get("llm_reason_fa"):
        print(f"{indent}{llm['llm_reason_fa']}")


def _print_llm_summary(result):
    """یک خط وضعیت لایه‌ی LLM برای خروجی اسکن (فقط وقتی با --llm اجرا شده)."""
    summary = result.get("llm_summary")
    if summary is None:
        return
    if not summary.get("active"):
        print("🧠 لایه‌ی LLM غیرفعال است: کلید NVIDIA تنظیم نشده "
              "(کلید را در config.NVIDIA_API_KEY بگذار یا متغیر محیطی "
              "NVIDIA_API_KEY را ست کن) — اسکن بدون بازبینی LLM ادامه یافت.")
    else:
        print(f"🧠 بازبینی LLM: {summary.get('analyzed', 0)} نامزد بررسی شد.")


def _run_llm_single(symbol, timeframe, source):
    """
    تحلیل عمیق تک‌نماد با --llm: همان امتیازدهی/لایه‌ی EV اسکنر روی همان
    نماد/تایم‌فریم، بعد بازبینی LLM (فقط اگر نماد از دروازه‌ی EV رد شود).
    اگر کلیدی تنظیم نشده باشد، یک خط واضح چاپ و خروجی عادی همین مسیر
    برمی‌گردد. روی هر خطای LLM، خروجی عادی بدون بلوک LLM چاپ می‌شود.
    """
    import scanner
    from analysis import ev as ev_mod
    market = "forex" if source in ("forex", "market") else None
    if market:
        from analysis.analyzer import fetch_market
        df = fetch_market(symbol, timeframe, market=market)
    else:
        df = fetch_candles_binance(symbol, timeframe)
    report = build_report(df, symbol, timeframe)
    cand = scanner.score_candidate(df, report, symbol, timeframe)
    scanner.attach_ev_decision(cand)
    print_report(report)
    print(f"\n{SEPARATOR}")
    print(f"تصمیم لایه‌ی EV: {cand['decision_fa']} ({cand['decision']})")
    print(ev_mod.format_ev_fa(cand.get("ev")))
    if cand["decision"] in ("NO_TRADE", "WAIT"):
        print(f"دلیل: {cand.get('decision_reason', '')}")
    scanner.attach_llm_analysis(cand)
    if "llm" not in cand:
        if cand.get("decision") != "NO_TRADE":
            print("🧠 لایه‌ی LLM غیرفعال است: کلید NVIDIA تنظیم نشده "
                  "(config.NVIDIA_API_KEY یا متغیر محیطی NVIDIA_API_KEY).")
        else:
            print("🧠 بازبینی LLM انجام نشد: کف سخت EV — نامزد NO_TRADE است "
                  "و LLM اجازه‌ی بازبینی رد اجباری را ندارد.")
    else:
        _print_llm_block(cand)
    print(cand["reason"])
    print("⚠️ بازبینی LLM نظر دوم استدلالی است، نه سیگنال تازه؛ سقف تصمیم را EV تعیین می‌کند.")


def _run_scan(llm=False):
    """اسکن بازار، چاپ جدول رتبه‌بندی و جزئیات پیشنهاد اول + نمودارش."""
    import scanner
    from analysis import ev as ev_mod
    print("\nدر حال اسکن بازار (منبع: بایننس) ...")
    result = scanner.scan_market(limit=config.SCAN_TOP_N, llm=llm)
    candidates = result["candidates"]

    if not candidates:
        print("\n❌ هیچ نامزدی اسکن نشد.")
        for e in result["errors"]:
            print(f"  • {e['symbol']} {e['timeframe']}: {e['error']}")
        return

    print("\n" + "=" * 50)
    print("  🏆 رتبه‌بندی اسکن بازار")
    print("=" * 50)
    for i, c in enumerate(candidates, 1):
        dec = f" | تصمیم: {c['decision_fa']}" if c.get("decision") else ""
        ev_part = _ev_part(c)
        print(f"{i}. {c['symbol']}/USDT | تایم‌فریم {c['timeframe']} | {c['direction_fa']} | "
              f"امتیاز: {c['score']}/۱۰۰ | قیمت: {_fmt_price(c['price'])}{dec}{ev_part}")
        _print_llm_block(c, indent="   ")
    summary = result.get("decision_summary") or {}
    if summary:
        order = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "WAIT", "NO_TRADE"]
        parts = [f"{ev_mod.DECISION_FA[d]}: {summary[d]}"
                 for d in order if summary.get(d)]
        print("خلاصه‌ی تصمیم‌ها (کل اسکن): " + " | ".join(parts))
        if sum(summary.get(d, 0) for d in ev_mod.ACTIONABLE_DECISIONS) == 0:
            print("امروز فرصت معتبر نیست — NO TRADE (هیچ نامزدی از دروازه‌ی EV عبور نکرد)")
    _print_llm_summary(result)

    top = candidates[0]
    comp = top["components"]
    print(f"\n{SEPARATOR}")
    if top.get("decision") == "NO_TRADE":
        print(f"رتبه‌ی اول اسکن (ردشده — NO TRADE): {top['symbol']}/USDT — "
              f"تایم‌فریم {top['timeframe']} — {top['direction_fa']} — امتیاز {top['score']}/۱۰۰")
    else:
        print(f"⭐ پیشنهاد امروز: {top['symbol']}/USDT — تایم‌فریم {top['timeframe']} — "
              f"{top['direction_fa']} — امتیاز {top['score']}/۱۰۰")
    print(SEPARATOR)
    print(f"جزئیات امتیاز: روند {comp['trend']}/{config.SCORE_TREND_MAX:.0f} | "
          f"مومنتوم {comp['momentum']}/{config.SCORE_MOMENTUM_MAX:.0f} | "
          f"الگو {comp['pattern']}/{config.SCORE_PATTERN_MAX:.0f} | "
          f"سودبه‌ریسک {comp['risk_reward']}/{config.SCORE_RR_MAX:.0f} | "
          f"حجم {comp['volume']}/{config.SCORE_VOLUME_MAX:.0f}")
    print(f"قیمت ورود: {_fmt_price(top['entry'])} | "
          f"حد ضرر: {_fmt_price(top['stop'])} | "
          f"هدف اول: {_fmt_price(top['target'])} | "
          f"نسبت سودبه‌ریسک: {top['risk_reward'] if top['risk_reward'] is not None else '—'}")
    if top["pattern"]:
        p = top["pattern"]
        rel = "" if p["reliable_sample"] else "  ⚠️ نمونه کم، آماری قابل اتکا نیست"
        print(f"الگوی پشتیبان: {p['fa']} — نرخ موفقیت تاریخی {p['success_rate_pct']}٪ "
              f"({p['total_occurrences']} رخداد روی همین ارز/تایم‌فریم){rel}")
    if top.get("decision"):
        print(f"تصمیم: {top['decision_fa']} ({top['decision']})")
        print(ev_mod.format_ev_fa(top.get("ev")))
        if top["decision"] in ("NO_TRADE", "WAIT"):
            print(f"دلیل: {top.get('decision_reason', '')}")
    print(top["reason"])
    _print_llm_block(top)
    print("⚠️ این پیشنهاد یک رتبه‌بندی مکانیکی است، نه تضمین سود. با مدیریت ریسک تصمیم بگیر.")

    cached = scanner.get_scan_cache(top["symbol"], top["timeframe"])
    if cached:
        df, report = cached
        try:
            _render_and_announce(df, report)
        except Exception as exc:
            print(f"⚠️ رسم نمودار پیشنهاد اول ناموفق بود: {exc}")

    if result["errors"]:
        print(f"\n⚠️ {len(result['errors'])} ترکیب (ارز/تایم‌فریم) به‌علت خطا از اسکن جا ماند.")


# ---------------------------------------------------------------------------
# منوی شماره‌دار (اجرای بدون آرگومان)
# ---------------------------------------------------------------------------
def _fa_digits_to_en(text: str) -> str:
    return text.translate(str.maketrans("۱۲۳۴۵۶۷۸۹۰", "1234567890"))


def _menu_single():
    symbol = input("Coin symbol (default BTC): ").strip() or "BTC"
    print("Timeframe:")
    print("1) 1h")
    print("2) 4h")
    print("3) 1d")
    print("4) 1w")
    choice = _fa_digits_to_en(input("Option (default 2): ").strip())
    tf_map = {"1": "1h", "2": "4h", "3": "1d", "4": "1w",
              "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
    timeframe = tf_map.get(choice, "4h")

    print(f"\nدر حال گرفتن داده و تحلیل {symbol}/USDT در تایم‌فریم {timeframe} (بایننس) ...")
    try:
        df, report = _analyze(symbol, config.DEFAULT_VS_CURRENCY, timeframe, "binance")
        print_report(report)
        _render_and_announce(df, report)
    except (AnalysisError, CoinGeckoError, BinanceError) as exc:
        print(f"\n❌ خطا: {exc}")
    except Exception as exc:
        print(f"\n❌ خطای غیرمنتظره: {exc}")


def _menu_paper():
    """زیرمنوی معاملات کاغذی (صرافی فرضی) — حساب CLI با کلید «cli».

    منطق در paper_trading.py است؛ اینجا فقط پوسته‌ی خط‌فرمان است و فقط
    حالت دستی دارد (بدون حالت خودکار). در ورود، موتور ارزیابی اجرا می‌شود.
    """
    import paper_trading as paper
    chat_id = "cli"
    try:
        for ev in paper.evaluate(chat_id):
            print(ev["message"])
    except Exception as exc:
        print(f"⚠️ evaluate failed: {exc}")
    while True:
        print("\n" + "=" * 50)
        print("  Paper Trading (fake exchange — live Binance prices)")
        print("=" * 50)
        print("1) View account & positions")
        print("2) Open position (manual)")
        print("3) Close position")
        print("4) Back to main menu")
        choice = _fa_digits_to_en(input("\nChoose an option (1-4): ").strip())
        try:
            if choice == "1":
                print(paper.summary_text(chat_id))
                print(paper.positions_text(chat_id))
            elif choice == "2":
                symbol = input("Symbol (default BTC): ").strip() or "BTC"
                print("Direction: 1) Long  2) Short")
                d = _fa_digits_to_en(input("Option (default 1): ").strip())
                direction = "short" if d == "2" else "long"
                amount = float(input("Amount in USDT: ").strip())
                stop_in = input("Stop loss price (blank = none): ").strip()
                target_in = input("Take profit price (blank = none): ").strip()
                pos = paper.open_position(
                    chat_id, symbol, direction, amount,
                    stop=float(stop_in) if stop_in else None,
                    target=float(target_in) if target_in else None)
                print(f"✅ Opened #{pos['id']} {pos['symbol']}/USDT {direction} | "
                      f"amount {pos['amount_usdt']:,.2f} USDT | entry {_fmt_price(pos['entry_price'])} | "
                      f"stop {_fmt_price(pos['stop'])} | target {_fmt_price(pos['target'])}")
            elif choice == "3":
                print(paper.positions_text(chat_id))
                pid = _fa_digits_to_en(input("Position id to close: ").strip())
                rec = paper.close_position(chat_id, int(pid))
                print(f"✅ Closed #{rec['id']} | exit {_fmt_price(rec['exit_price'])} | "
                      f"net PnL {rec['net_pnl']:+,.2f} USDT")
            else:
                return
        except Exception as exc:
            print(f"❌ Error: {exc}")


def _run_forex_scan(llm=False):
    """اسکن فارکس (افزودنی): همان قالب خروجی اسکن کریپتو، بازار فارکس."""
    import scanner
    from analysis import ev as ev_mod
    print("\nدر حال اسکن بازار فارکس (منبع: Yahoo Finance — پشتیبان: Dukascopy) ...")
    print("نکته: XAUUSD طلا (نماینده‌ی کالا) است، نه جفت‌ارز؛ قیمت‌ها از "
          "فید تاریخی با تأخیر می‌آیند (FOREX_FA.md).")
    result = scanner.scan_forex_market(limit=config.SCAN_TOP_N, llm=llm)
    candidates = result["candidates"]

    if not candidates:
        print("\n❌ هیچ نامزدی اسکن نشد.")
        for e in result["errors"]:
            print(f"  • {e['symbol']} {e['timeframe']}: {e['error']}")
        return

    print("\n" + "=" * 50)
    print("  🏆 رتبه‌بندی اسکن فارکس")
    print("=" * 50)
    for i, c in enumerate(candidates, 1):
        dec = f" | تصمیم: {c['decision_fa']}" if c.get("decision") else ""
        ev_part = _ev_part(c)
        print(f"{i}. {c['symbol']} | تایم‌فریم {c['timeframe']} | {c['direction_fa']} | "
              f"امتیاز: {c['score']}/۱۰۰ | قیمت: {_fmt_price(c['price'])}{dec}{ev_part}")
        _print_llm_block(c, indent="   ")
    summary = result.get("decision_summary") or {}
    if summary:
        order = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "WAIT", "NO_TRADE"]
        parts = [f"{ev_mod.DECISION_FA[d]}: {summary[d]}"
                 for d in order if summary.get(d)]
        print("خلاصه‌ی تصمیم‌ها (کل اسکن): " + " | ".join(parts))
        if sum(summary.get(d, 0) for d in ev_mod.ACTIONABLE_DECISIONS) == 0:
            print("امروز فرصت معتبر فارکس نیست — NO TRADE")
    _print_llm_summary(result)

    top = candidates[0]
    print(f"\n{SEPARATOR}")
    print(f"رتبه‌ی اول فارکس: {top['symbol']} — تایم‌فریم {top['timeframe']} — "
          f"{top['direction_fa']} — امتیاز {top['score']}/۱۰۰")
    print(SEPARATOR)
    print(f"قیمت ورود: {_fmt_price(top['entry'])} | "
          f"حد ضرر: {_fmt_price(top['stop'])} | "
          f"هدف اول: {_fmt_price(top['target'])} | "
          f"نسبت سودبه‌ریسک: {top['risk_reward'] if top['risk_reward'] is not None else '—'}")
    if top.get("decision"):
        print(f"تصمیم: {top['decision_fa']} ({top['decision']})")
        print(ev_mod.format_ev_fa(top.get("ev")))
        if top["decision"] in ("NO_TRADE", "WAIT"):
            print(f"دلیل: {top.get('decision_reason', '')}")
    print(top["reason"])
    _print_llm_block(top)
    print("⚠️ این پیشنهاد یک رتبه‌بندی مکانیکی است، نه تضمین سود. با مدیریت ریسک تصمیم بگیر.")

    cached = scanner.get_scan_cache(top["symbol"], top["timeframe"])
    if cached:
        df, report = cached
        try:
            _render_and_announce(df, report)
        except Exception as exc:
            print(f"⚠️ رسم نمودار پیشنهاد اول ناموفق بود: {exc}")

    if result["errors"]:
        print(f"\n⚠️ {len(result['errors'])} ترکیب (نماد/تایم‌فریم) به‌علت خطا از اسکن جا ماند.")


def _menu():
    print("\n" + "=" * 50)
    print("  Crypto Technical Analyzer (v2)")
    print("=" * 50)
    print("1) Analyze one coin")
    print("2) Market scan + top pick")
    print("3) Exit")
    print("4) Paper trading (fake exchange)")
    print("5) Forex scan (اسکن فارکس)")
    choice = _fa_digits_to_en(input("\nChoose an option (1-5): ").strip())
    if choice == "1":
        _menu_single()
    elif choice == "2":
        _run_scan()
    elif choice == "4":
        _menu_paper()
    elif choice == "5":
        _run_forex_scan()
    else:
        print("Bye.")


def main():
    if len(sys.argv) == 1:
        _menu()
        return

    parser = argparse.ArgumentParser(description="ابزار تحلیل تکنیکال ارز دیجیتال")
    parser.add_argument("--symbol", "-s", help="نماد یا نام ارز (مثلاً btc یا bitcoin)")
    parser.add_argument("--vs", default=config.DEFAULT_VS_CURRENCY, help="ارز پایه (پیش‌فرض: usd)")
    parser.add_argument("--timeframe", "-t", default="1d",
                        choices=list(config.TIMEFRAME_DAYS_MAP.keys()),
                        help="تایم‌فریم تحلیل")
    parser.add_argument("--source", default="binance", choices=["binance", "coingecko", "forex", "market"],
                        help="منبع داده (پیش‌فرض: binance؛ coingecko مسیر قدیمی است؛ forex/market برای فارکس)")
    parser.add_argument("--scan", action="store_true",
                        help="اسکن بازار روی چند ارز و پیشنهاد بهترین نامزد")
    parser.add_argument("--scan-forex", action="store_true",
                        help="اسکن بازار فارکس (Dukascopy) و پیشنهاد بهترین نامزد")
    parser.add_argument("--chart", action="store_true",
                        help="ذخیره‌ی تصویر نمودار تحلیل (PNG)")
    parser.add_argument("--paper", action="store_true",
                        help="منوی معاملات کاغذی (صرافی فرضی با قیمت زنده)")
    parser.add_argument("--refresh-ev", action="store_true",
                        help="بازنشانی کش EV: بک‌تست تازه برای همه‌ی ترکیب‌های اسکن و ذخیره در ev_cache.json (کند است)")
    parser.add_argument("--llm", action="store_true",
                        help="بازبینی LLM (NVIDIA) روی خروجی همین اجرا: برای --scan/--scan-forex روی حداکثر ۳ نامزد اولِ غیر NO_TRADE و برای تحلیل تک‌نماد روی همان نامزد؛ بدون کلید NVIDIA فقط یک خط وضعیت چاپ می‌شود و خروجی عادی برمی‌گردد (LLM_LAYER_FA.md)")
    args = parser.parse_args()

    if args.refresh_ev:
        from analysis import ev as ev_mod
        print("در حال بازنشانی کش EV (برای هر ترکیب یک بک‌تست کامل اجرا می‌شود) ...")
        res = ev_mod.refresh_ev_cache(config.SCAN_SYMBOLS, config.SCAN_TIMEFRAMES)
        for d in res["refreshed"]:
            print(f"  • {d['symbol']} {d['timeframe']}: {d['n_trades']} معامله‌ی خارج‌ازنمونه | EV {d['expectancy_r']}R")
        for e in res["errors"]:
            print(f"  ⚠️ {e['symbol']} {e['timeframe']}: {e['error']}")
        print(f"تمام شد: {len(res['refreshed'])} ترکیب بازنشانی، {len(res['errors'])} خطا.")
        return

    if args.scan:
        _run_scan(llm=args.llm)
        return

    if args.scan_forex:
        _run_forex_scan(llm=args.llm)
        return

    if args.paper:
        _menu_paper()
        return

    symbol = args.symbol
    if not symbol:
        symbol = input("نماد یا نام ارز را وارد کن (مثلاً btc): ").strip()
        if not symbol:
            print("نمادی وارد نشد. خروج.")
            sys.exit(1)

    print(f"\nدر حال گرفتن داده و تحلیل {symbol} در تایم‌فریم {args.timeframe} "
          f"(منبع: {args.source}) ...")
    try:
        _df = _report = None
        if args.llm:
            _run_llm_single(symbol, args.timeframe, args.source)
        else:
            _df, _report = _analyze(symbol, args.vs, args.timeframe, args.source)
            print_report(_report)
        if args.chart:
            if _df is None:
                _df, _report = _analyze(symbol, args.vs, args.timeframe, args.source)
            _render_and_announce(_df, _report)
    except (AnalysisError, CoinGeckoError, BinanceError) as exc:
        print(f"\n❌ خطا: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ خطای غیرمنتظره: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
