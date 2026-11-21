#!/usr/bin/env python3
"""
telegram_bot.py — Telegram interface for the technical-analysis tool (version 2)

The core logic (Binance feed, indicators, patterns, backtest, and scanner) is
unchanged; this file is only a Telegram shell over the same functions.

Run:  python telegram_bot.py

The buttons are "glass": in Telegram, the inline buttons are the glass buttons.
The scan is live: during the scan, the status message is continuously edited with
a progress bar and the current symbol/timeframe, so the bot does not look "dead"
while scanning.
"""

import asyncio
import html
import io
import logging
import os
from contextlib import redirect_stdout
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (ApplicationBuilder, CallbackQueryHandler,
                          CommandHandler, ContextTypes, MessageHandler,
                          filters)

import config
import scanner as _scanner_mod
import paper_trading as paper
from analysis import ev as ev_mod


def _ev_lines(c) -> list:
    """Display lines for the EV layer for a candidate (decision + statistics + rejection reason)."""
    if not c.get("decision"):
        return []
    lines = [f"تصمیم: {c['decision_fa']} ({c['decision']})",
             ev_mod.format_ev_fa(c.get("ev"))]
    if c.get("decision") in ("NO_TRADE", "WAIT"):
        lines.append(f"دلیل: {c.get('decision_reason', '')}")
    return lines


def _llm_lines(c) -> list:
    """Compact LLM-review block for a candidate — only when it has an "llm" key.

    Without an llm key, an empty list is returned and the message text remains exactly
    as before. The bot itself never calls the LLM layer (each review can cost/time up
    to 5 reasoning-model calls); this function only displays a result that another
    path (the CLI with --llm or a programmatic scan) attached to the candidate.
    Details: LLM_LAYER_FA.md.
    """
    llm = c.get("llm")
    if not llm:
        return []
    votes = llm.get("llm_votes") or {}
    votes_txt = "، ".join(f"{k}: {v}" for k, v in votes.items()) or "—"
    conf = llm.get("llm_confidence")
    lines = [f"🧠 بازبینی LLM: تصمیم {llm.get('llm_decision')} | "
             f"اطمینان {f'{conf}٪' if conf is not None else '—'} | آرا: {votes_txt}"]
    if llm.get("llm_crosscheck_disagreement"):
        lines.append("⚠️ کراس‌چک مستقل با اکثریت مخالف بود (یک پله تنزل اعمال شد).")
    if llm.get("llm_reason_fa"):
        lines.append(llm["llm_reason_fa"])
    return lines


# Single-symbol suggestion: the scanner's scoring is run on the same symbol/timeframe.
def _verdict(symbol, timeframe, df, report) -> str:
    c = _scanner_mod.score_candidate(df, report, symbol, timeframe)
    try:
        _scanner_mod.attach_ev_decision(c)
    except Exception:
        pass
    comp = c["components"]
    lines = [
        "═══ 🧭 پیشنهاد برای همین ارز ═══",
        f"{c['symbol']}/USDT — {c['timeframe']} — {c['direction_fa']} — امتیاز کلی {c['score']}/100",
        f"امتیازها: روند {comp['trend']}/{config.SCORE_TREND_MAX:.0f} | "
        f"مومنتوم {comp['momentum']}/{config.SCORE_MOMENTUM_MAX:.0f} | "
        f"الگو {comp['pattern']}/{config.SCORE_PATTERN_MAX:.0f} | "
        f"سودبه‌ریسک {comp['risk_reward']}/{config.SCORE_RR_MAX:.0f} | "
        f"حجم {comp['volume']}/{config.SCORE_VOLUME_MAX:.0f}",
        f"ورود: {_fmt_price(c['entry'])} | حد ضرر: {_fmt_price(c['stop'])} | "
        f"هدف اول: {_fmt_price(c['target'])} | R/R: {c['risk_reward'] if c['risk_reward'] is not None else '—'}",
    ]
    if c.get("pattern"):
        p = c["pattern"]
        rel = "" if p["reliable_sample"] else "  ⚠️ نمونه کم"
        lines.append(f"الگوی پشتیبان: {p['fa']} — موفقیت تاریخی {p['success_rate_pct']}٪ "
                     f"({p['total_occurrences']} رخداد){rel}")
    # Step 5: derivatives context (funding/open interest) — display only, and only if
    # data is returned; get_derivatives_context returns None on any error.
    # Forex symbols have no Binance derivatives, so this section is skipped for them.
    from data import forex_client as _fx
    dctx = None
    if not _fx.is_forex_symbol(symbol):
        try:
            from data.futures_client import get_derivatives_context
            dctx = get_derivatives_context(symbol)
        except Exception:
            dctx = None
    if dctx:
        oi_part = ""
        if dctx.get("open_interest") is not None:
            oi_part = f" | بهره باز: {dctx['open_interest']:,.0f}"
            if dctx.get("oi_change_24h_pct") is not None:
                oi_part += f" ({dctx['oi_change_24h_pct']:+.2f}٪ در ۲۴ ساعت)"
        deriv_line = (f"📈 مشتقه: فاندینگ {dctx['funding_rate_pct']:+.4f}٪ — "
                      f"{dctx['funding_bias_fa']}{oi_part}")
        # Interpretation layer (additive): funding label+note and the price×open-interest quadrant
        fi = dctx.get("funding_interpretation")
        if fi:
            deriv_line += f" | تفسیر فاندینگ: {fi['fa']} — {fi['note']}"
        oi_i = dctx.get("oi_interpretation")
        if oi_i:
            deriv_line += f" | قیمت×بهره‌باز: {oi_i['fa']}"
        lines.append(deriv_line)
    # Step 6: suggested position size (placeholder account — stated in the text)
    sz = c.get("sizing")
    if sz:
        lines.append(f"حجم پیشنهادی (ریسک {sz['risk_pct']:.0f}٪ از حساب نمونه "
                     f"{sz['account_size_usdt']:,.0f} تتری): {sz['size_units']} واحد "
                     f"≈ {sz['notional_usdt']:,.2f} USDT")
    lines.extend(_ev_lines(c))
    lines.extend(_llm_lines(c))
    lines.append(c["reason"])
    lines.append("⚠️ رتبه‌بندی مکانیکی است، نه تضمین سود.")
    return "\n".join(lines)

# Telegram only uses api.telegram.org; this environment's no_proxy may contain
# bracketed IPv6 values that break httpx — we restrict it.
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
from analysis.analyzer import build_report, fetch_candles_binance, fetch_market
import charting
import scanner
from main import _fmt_price, print_report

# ---------------------------------------------------------------------------
# Settings — keep the token here (security suggestion: after testing, reset the bot in
# BotFather with /revoke and put the new token only here)
# ---------------------------------------------------------------------------
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # get your own bot token from @BotFather "YOUR_BOT_TOKEN"
ALLOWED_USER_IDS = []          # Empty = everyone; to lock it to yourself, put your numeric ID here
TIMEFRAMES = ["1h", "4h", "1d", "1w"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto_tg_bot")

# ---------------------------------------------------------------------------
# Glassmorphism (inline) keyboards
# ---------------------------------------------------------------------------
def kb_main():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 تحلیل یک ارز", callback_data="m:analyze"),
         InlineKeyboardButton("🔍 اسکن بازار", callback_data="m:scan")],
        [InlineKeyboardButton("💱 اسکن فارکس", callback_data="m:fxscan")],
        [InlineKeyboardButton("🧪 معاملات کاغذی", callback_data="p:menu")],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data="m:help")],
    ])


# --- Paper-trading keyboards (hypothetical exchange) ---
def kb_paper(chat_id):
    """Paper-trading submenu; manual opening is the first row and auto mode sits beside it."""
    try:
        auto_on = paper.get_account(chat_id)["auto"]["enabled"]
    except Exception:
        auto_on = False
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ باز کردن پوزیشن (دستی)", callback_data="p:open"),
         InlineKeyboardButton("🤖 خودکار: " + ("روشن ✅" if auto_on else "خاموش"),
                              callback_data="p:auto")],
        [InlineKeyboardButton("💰 موجودی و سبد", callback_data="p:summary"),
         InlineKeyboardButton("📂 پوزیشن‌های باز", callback_data="p:positions")],
        [InlineKeyboardButton("📜 تاریخچه", callback_data="p:history"),
         InlineKeyboardButton("♻️ ریست حساب", callback_data="p:reset")],
        [InlineKeyboardButton("🏠 منوی اصلی", callback_data="m:home")],
    ])


def kb_paper_positions(chat_id):
    """List of positions with a close button for each position (manual and automatic together)."""
    rows = []
    try:
        for p in paper.list_positions(chat_id):
            src = "🤖" if p.get("source") == "auto" else "✋"
            rows.append([InlineKeyboardButton(
                f"❌ بستن #{p['id']} {src} {p['symbol']} ({p['net_pnl']:+,.2f})",
                callback_data=f"p:close:{p['id']}")])
    except Exception:
        pass
    rows.append([InlineKeyboardButton("🔙 منوی کاغذی", callback_data="p:menu")])
    return InlineKeyboardMarkup(rows)


def kb_paper_direction():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🟢 لانگ (خرید)", callback_data="p:dir:long"),
         InlineKeyboardButton("🔴 شورت (فروش)", callback_data="p:dir:short")],
        [InlineKeyboardButton("🔙 منوی کاغذی", callback_data="p:menu")],
    ])


def kb_paper_amount():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("۵۰ USDT", callback_data="p:amt:50"),
         InlineKeyboardButton("۱۰۰ USDT", callback_data="p:amt:100"),
         InlineKeyboardButton("۲۵۰ USDT", callback_data="p:amt:250")],
        [InlineKeyboardButton("✏️ مبلغ دلخواه", callback_data="p:amt:custom")],
        [InlineKeyboardButton("🔙 منوی کاغذی", callback_data="p:menu")],
    ])


def kb_paper_mode():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🤖 استاپ و هدف از تحلیل ربات", callback_data="p:mode:auto")],
        [InlineKeyboardButton("🚫 بدون استاپ/هدف", callback_data="p:mode:none"),
         InlineKeyboardButton("✏️ دستی وارد می‌کنم", callback_data="p:mode:manual")],
        [InlineKeyboardButton("🔙 منوی کاغذی", callback_data="p:menu")],
    ])


def kb_timeframes(symbol):
    row = [InlineKeyboardButton(tf, callback_data=f"tf:{symbol}:{tf}")
           for tf in TIMEFRAMES]
    return InlineKeyboardMarkup(
        [row, [InlineKeyboardButton("🔙 منوی اصلی", callback_data="m:home")]])


def kb_after_analysis(symbol, timeframe):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖼 نمودار همین تحلیل", callback_data=f"ch:{symbol}:{timeframe}"),
         InlineKeyboardButton("🔍 اسکن بازار", callback_data="m:scan")],
        [InlineKeyboardButton("📊 تحلیل ارز دیگر", callback_data="m:analyze"),
         InlineKeyboardButton("🏠 منوی اصلی", callback_data="m:home")],
    ])


def kb_after_scan():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖼 نمودار پیشنهاد اول", callback_data="ch:top")],
        [InlineKeyboardButton("🔁 اسکن دوباره", callback_data="m:scan"),
         InlineKeyboardButton("🏠 منوی اصلی", callback_data="m:home")],
    ])


WELCOME = ("📈 <b>ربات تحلیل تکنیکال کریپتو</b>\n\n"
           "یک گزینه را انتخاب کن، یا نماد ارز را مستقیم بفرست (مثل BTC):")
HELP = ("<b>راهنما</b>\n\n"
        "• «تحلیل یک ارز»: نماد را می‌فرستی و تایم‌فریم را با دکمه انتخاب می‌کنی.\n"
        "• «اسکن بازار»: ۲۰ ارز × دو تایم‌فریم اسکن و رتبه‌بندی می‌شود و پیشنهاد اول با جزئیات می‌آید.\n"
        "• دکمه‌ی «نمودار» تصویر چارت با سطوح و حد ضرر را می‌فرستد.\n"
        "• میان‌بر: نماد را با تایم‌فریم بفرست، مثل: <code>ETH 4h</code>\n"
        "• «معاملات کاغذی»: صرافی فرضی با قیمت زنده‌ی بایننس و موجودی مجازی؛ "
        "پوزیشن را دستی باز/بسته کن (مبلغ، جهت و حد ضرر/هدف دلخواه) یا حالت "
        "خودکار را روشن کن تا ربات با امتیازهای خودش معامله‌ی کاغذی باز کند. "
        "پوزیشن‌های دستی ✋ و خودکار 🤖 در یک حساب کنار هم‌اند.\n\n"
        "⚠️ خروجی‌ها رتبه‌بندی مکانیکی‌اند، نه تضمین سود.")


def _allowed(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)


def _render_report(symbol, timeframe, df, report) -> str:
    """Convert the existing text report (print_report) to a string without changing its logic."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(report)
    return buf.getvalue()


def _chart_bytes(df, report) -> bytes:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{report['symbol'].upper()}_{report['timeframe']}_{ts}.png"
    path = charting.render_chart(df, report, os.path.join(config.CHART_OUTPUT_DIR, name))
    with open(path, "rb") as fh:
        return fh.read()


def _scan_detail_text(top) -> str:
    comp = top["components"]
    heading = ("═══ ⭐ پیشنهاد اول ═══" if top.get("decision") != "NO_TRADE"
               else "═══ رتبه‌ی اول اسکن (ردشده — NO TRADE) ═══")
    lines = [
        heading,
        f"{top['symbol']}/USDT — {top['timeframe']} — {top['direction_fa']} — امتیاز {top['score']}/100",
        f"امتیازها: روند {comp['trend']}/{config.SCORE_TREND_MAX:.0f} | "
        f"مومنتوم {comp['momentum']}/{config.SCORE_MOMENTUM_MAX:.0f} | "
        f"الگو {comp['pattern']}/{config.SCORE_PATTERN_MAX:.0f} | "
        f"سودبه‌ریسک {comp['risk_reward']}/{config.SCORE_RR_MAX:.0f} | "
        f"حجم {comp['volume']}/{config.SCORE_VOLUME_MAX:.0f}",
        f"ورود: {_fmt_price(top['entry'])} | حد ضرر: {_fmt_price(top['stop'])} | "
        f"هدف اول: {_fmt_price(top['target'])} | R/R: {top['risk_reward'] if top['risk_reward'] is not None else '—'}",
    ]
    if top.get("pattern"):
        p = top["pattern"]
        rel = "" if p["reliable_sample"] else "  ⚠️ نمونه کم"
        lines.append(f"الگوی پشتیبان: {p['fa']} — موفقیت تاریخی {p['success_rate_pct']}٪ "
                     f"({p['total_occurrences']} رخداد){rel}")
    lines.extend(_ev_lines(top))
    lines.extend(_llm_lines(top))
    lines.append(top["reason"])
    lines.append("⚠️ رتبه‌بندی مکانیکی است، نه تضمین سود.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.HTML,
                                    reply_markup=kb_main())


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text(HELP, parse_mode=ParseMode.HTML,
                                    reply_markup=kb_main())


async def _do_analysis(chat, symbol, timeframe, status_msg=None):
    """Analysis + sending the text report; returns (df, report) so the chart button works.

    Forex additive: forex symbols are fetched through the market route (Yahoo;
    Dukascopy fallback); everything else comes from Binance exactly as before.
    """
    from data import forex_client
    fetch = (lambda s, t: fetch_market(s, t, market="forex")) \
        if forex_client.is_forex_symbol(symbol) else fetch_candles_binance
    df, report = await asyncio.to_thread(
        lambda: (lambda d: (d, build_report(d, symbol, timeframe)))(
            fetch(symbol, timeframe)))
    text = _render_report(symbol, timeframe, df, report)
    if status_msg is not None:
        try:
            await status_msg.delete()
        except Exception:
            pass
    for i in range(0, len(text), 3900):
        await chat.send_message(f"<pre>{html.escape(text[i:i + 3900])}</pre>",
                                parse_mode=ParseMode.HTML)
    verdict = _verdict(symbol, timeframe, df, report)
    await chat.send_message(f"<pre>{html.escape(verdict)}</pre>",
                            parse_mode=ParseMode.HTML,
                            reply_markup=kb_after_analysis(symbol, timeframe))
    return df, report


async def _do_scan(chat, status_msg):
    """Live scan: the combination count is read from the scanner itself (fetch+score in one step).

    Note: the internal scanner records an error for each combination and continues, but
    if an error occurs in the middle of fetch and before score_candidate (for example,
    a Binance timeout), that combination is not counted. So progress is read from
    get_scan_cache, and at the end the difference is reported as the errored items.
    The core scanner is untouched.
    """
    symbols, timeframes = config.SCAN_SYMBOLS, config.SCAN_TIMEFRAMES
    total = len(symbols) * len(timeframes)
    base = len(scanner._SCAN_CACHE)
    state = {"fin": False}

    async def ticker():
        while not state["fin"]:
            done = max(0, len(scanner._SCAN_CACHE) - base)
            filled = int(done / total * 14) if total else 0
            bar = "▰" * filled + "▱" * (14 - filled)
            try:
                await status_msg.edit_text(f"🔍 در حال اسکن بازار...\n{bar} {done}/{total}")
            except Exception:
                pass
            await asyncio.sleep(1.2)

    tick = asyncio.create_task(ticker())
    try:
        res = await asyncio.to_thread(scanner.scan_market, None, None, config.SCAN_TOP_N, False)
    finally:
        state["fin"] = True
        tick.cancel()
    return res, total


async def _run_scan_flow(chat, status_msg):
    result, total = await _do_scan(chat, status_msg)
    failed = total - result["scanned"]
    candidates = result["candidates"]
    try:
        await status_msg.delete()
    except Exception:
        pass
    if not candidates:
        await chat.send_message("❌ نامزدی اسکن نشد.", reply_markup=kb_main())
        return
    lines = ["🏆 <b>رتبه‌بندی اسکن بازار</b>"]
    for i, c in enumerate(candidates, 1):
        dec = f" | تصمیم: {c['decision_fa']}" if c.get("decision") else ""
        lines.append(f"{i}. {c['symbol']}/USDT | {c['timeframe']} | {c['direction_fa']} | "
                     f"امتیاز {c['score']}/100 | {_fmt_price(c['price'])}{dec}")
    summary = result.get("decision_summary") or {}
    if summary:
        order = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "WAIT", "NO_TRADE"]
        parts = [f"{ev_mod.DECISION_FA[d]}: {summary[d]}"
                 for d in order if summary.get(d)]
        lines.append("خلاصه‌ی تصمیم‌ها (کل اسکن): " + " | ".join(parts))
        actionable = sum(summary.get(d, 0) for d in ev_mod.ACTIONABLE_DECISIONS)
        if actionable == 0:
            lines.append("امروز فرصت معتبر نیست — NO TRADE "
                         "(هیچ نامزدی از دروازه‌ی EV عبور نکرد)")
    await chat.send_message("\n".join(lines), parse_mode=ParseMode.HTML)
    top = candidates[0]
    await chat.send_message(f"<pre>{html.escape(_scan_detail_text(top))}</pre>",
                            parse_mode=ParseMode.HTML, reply_markup=kb_after_scan())
    cached = scanner.get_scan_cache(top["symbol"], top["timeframe"])
    if cached:
        context_cache["top"] = cached
    if failed or result["errors"]:
        err_lines = "\n".join(f"• {e['symbol']} {e['timeframe']}: {e['error']}"
                              for e in result["errors"][:8])
        await chat.send_message(f"⚠️ {max(failed, len(result['errors']))} ترکیب در اسکن خطا خورد "
                                f"(گذرا/شبکه‌ای؛ اسکن تکراری معمولاً پاک می‌شود):\n{err_lines}",
                                reply_markup=kb_main())


async def _run_forex_scan_flow(chat, status_msg):
    """Forex scan (additive): mirrors the crypto scan flow over scan_forex_market."""
    total = len(config.FOREX_SYMBOLS) * len(config.FOREX_TIMEFRAMES)
    try:
        result = await asyncio.to_thread(
            scanner.scan_forex_market, None, None, config.SCAN_TOP_N, False)
    finally:
        try:
            await status_msg.delete()
        except Exception:
            pass
    candidates = result["candidates"]
    if not candidates:
        err = "\n".join(f"• {e['symbol']} {e['timeframe']}: {e['error']}"
                        for e in result["errors"][:8])
        await chat.send_message(f"❌ نامزد فارکسی اسکن نشد.\n{err}",
                                reply_markup=kb_main())
        return
    lines = ["💱 <b>رتبه‌بندی اسکن فارکس</b> (منبع: Yahoo Finance — قیمت‌ها با تأخیر؛ "
             "XAUUSD طلاست)"]
    for i, c in enumerate(candidates, 1):
        dec = f" | تصمیم: {c['decision_fa']}" if c.get("decision") else ""
        lines.append(f"{i}. {c['symbol']} | {c['timeframe']} | {c['direction_fa']} | "
                     f"امتیاز {c['score']}/100 | {_fmt_price(c['price'])}{dec}")
    summary = result.get("decision_summary") or {}
    if summary:
        order = ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL", "WAIT", "NO_TRADE"]
        parts = [f"{ev_mod.DECISION_FA[d]}: {summary[d]}"
                 for d in order if summary.get(d)]
        lines.append("خلاصه‌ی تصمیم‌ها (کل اسکن): " + " | ".join(parts))
        actionable = sum(summary.get(d, 0) for d in ev_mod.ACTIONABLE_DECISIONS)
        if actionable == 0:
            lines.append("امروز فرصت معتبر فارکس نیست — NO TRADE "
                         "(هیچ نامزدی از دروازه‌ی EV عبور نکرد)")
    await chat.send_message("\n".join(lines), parse_mode=ParseMode.HTML)
    top = candidates[0]
    detail = _scan_detail_text(top).replace("/USDT", "")
    await chat.send_message(f"<pre>{html.escape(detail)}</pre>",
                            parse_mode=ParseMode.HTML, reply_markup=kb_after_scan())
    cached = scanner.get_scan_cache(top["symbol"], top["timeframe"])
    if cached:
        context_cache["top"] = cached
    if result["errors"]:
        err_lines = "\n".join(f"• {e['symbol']} {e['timeframe']}: {e['error']}"
                              for e in result["errors"][:8])
        await chat.send_message(f"⚠️ {len(result['errors'])} ترکیب در اسکن فارکس خطا خورد:\n{err_lines}",
                                reply_markup=kb_main())


context_cache = {}


# ---------------------------------------------------------------------------
# Paper trading — the logic is in paper_trading.py; this file is only the shell.
# Manual and automatic positions share one account, and each position carries an origin (source) label.
# ---------------------------------------------------------------------------
def _paper_stop_target(symbol, direction):
    """
    Stop/target from the bot's analysis (4h timeframe): the suggested stop loss for
    the same direction from the report + the first opposing level (resistance for
    long, support for short) as the target — exactly the same logic used in the
    "suggestion" text.
    """
    df = fetch_candles_binance(symbol, "4h")
    report = build_report(df, symbol, "4h")
    key = "stop_loss_long" if direction == "long" else "stop_loss_short"
    stop = ((report.get(key) or {}).get("recommended") or {}).get("stop_loss")
    sr = report.get("support_resistance") or {}
    levels = sr.get("resistance") if direction == "long" else sr.get("support")
    target = levels[0]["level"] if levels else None
    return stop, target


async def _paper_finish_open(update_or_query, context, chat):
    """Final step of the manual-open flow: execute the opening and announce the result."""
    flow = context.user_data.get("paper_flow") or {}
    symbol, direction = flow.get("symbol"), flow.get("direction")
    amount, mode = flow.get("amount"), flow.get("mode")
    stop = target = None
    try:
        if mode == "auto":
            await chat.send_message("⏳ در حال گرفتن استاپ/هدف از تحلیل ربات (4h) ...")
            stop, target = await asyncio.to_thread(_paper_stop_target, symbol, direction)
        elif mode == "manual":
            stop, target = flow.get("stop"), flow.get("target")
        pos = await asyncio.to_thread(
            paper.open_position, chat.id, symbol, direction, amount, stop, target)
    except Exception as exc:
        await chat.send_message(f"❌ بازکردن پوزیشن ناموفق بود: {exc}",
                                reply_markup=kb_paper(chat.id))
        return
    context.user_data.pop("paper_flow", None)
    await chat.send_message(
        f"✅ پوزیشن دستی ✋ باز شد:\n"
        f"#{pos['id']} {pos['symbol']}/USDT — {pos['direction']} | "
        f"مبلغ: {pos['amount_usdt']:,.2f} USDT | ورود: {_fmt_price(pos['entry_price'])}\n"
        f"حد ضرر: {_fmt_price(pos['stop'])} | هدف: {_fmt_price(pos['target'])}",
        reply_markup=kb_paper(chat.id))


async def _paper_show_positions(chat):
    await chat.send_message(paper.positions_text(chat.id),
                            reply_markup=kb_paper_positions(chat.id))


async def on_paper_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    query = update.callback_query
    chat = query.message.chat
    cid = chat.id

    # On every interaction, the evaluation engine runs first (so stop/target closures are seen)
    if data in ("p:menu", "p:summary", "p:positions", "p:history"):
        try:
            events = await asyncio.to_thread(paper.evaluate, cid)
            for ev in events:
                await chat.send_message(ev["message"])
        except Exception:
            pass

    if data == "p:menu":
        await query.message.reply_text("🧪 <b>معاملات کاغذی</b> — صرافی فرضی با قیمت زنده",
                                       parse_mode=ParseMode.HTML, reply_markup=kb_paper(cid))
        return
    if data == "p:summary":
        text = await asyncio.to_thread(paper.summary_text, cid)
        await query.message.reply_text(text, reply_markup=kb_paper(cid))
        return
    if data == "p:positions":
        await _paper_show_positions(chat)
        return
    if data == "p:history":
        await query.message.reply_text(paper.history_text(cid), reply_markup=kb_paper(cid))
        return
    if data == "p:auto":
        acct = paper.get_account(cid)
        new_state = not acct["auto"]["enabled"]
        paper.set_auto(cid, enabled=new_state)
        await query.message.reply_text(
            f"🤖 حالت خودکار {'روشن شد' if new_state else 'خاموش شد'} "
            f"(تایم‌فریم {acct['auto']['timeframe']}، آستانه {acct['auto']['threshold']:.0f}، "
            f"ریسک {acct['auto']['risk_pct']:.0f}٪، حداکثر {acct['auto']['max_open']} پوزیشن). "
            "پوزیشن‌های خودکار با 🤖 از دستی‌ها جدا مشخص می‌شوند. "
            "حالت خودکار علاوه بر آستانه‌ی امتیاز، دروازه‌ی EV هم دارد: فقط "
            "تصمیم‌های خرید/فروش با EV خالص مثبت باز می‌شوند؛ بازکردن دستی "
            "بدون دروازه است.",
            reply_markup=kb_paper(cid))
        return
    if data == "p:reset":
        await query.message.reply_text(
            "♻️ حساب کاغذی کامل ریست شود؟ موجودی به "
            f"{config.PAPER_START_BALANCE_USDT:,.0f} USDT برمی‌گردد و تاریخچه پاک می‌شود.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ بله، ریست شود", callback_data="p:reset:yes"),
                 InlineKeyboardButton("🔙 انصراف", callback_data="p:menu")]]))
        return
    if data == "p:reset:yes":
        paper.reset_account(cid)
        await query.message.reply_text("♻️ حساب ریست شد.", reply_markup=kb_paper(cid))
        return
    if data == "p:open":
        context.user_data["paper_flow"] = {}
        context.user_data["paper_await"] = "symbol"
        await query.message.reply_text("➕ نماد ارز را بفرست (مثل BTC یا ETH):")
        return
    if data.startswith("p:dir:"):
        flow = context.user_data.setdefault("paper_flow", {})
        flow["direction"] = data.split(":", 2)[2]
        await query.message.reply_text("مبلغ پوزیشن (USDT) را انتخاب کن:",
                                       reply_markup=kb_paper_amount())
        return
    if data.startswith("p:amt:"):
        flow = context.user_data.setdefault("paper_flow", {})
        val = data.split(":", 2)[2]
        if val == "custom":
            context.user_data["paper_await"] = "amount"
            await query.message.reply_text("مبلغ دلخواه را به USDT بفرست (مثل 150):")
            return
        flow["amount"] = float(val)
        await query.message.reply_text("حد ضرر و هدف چطور باشد؟", reply_markup=kb_paper_mode())
        return
    if data.startswith("p:mode:"):
        flow = context.user_data.setdefault("paper_flow", {})
        flow["mode"] = data.split(":", 2)[2]
        if flow["mode"] == "manual":
            context.user_data["paper_await"] = "stop"
            await query.message.reply_text(
                "قیمت حد ضرر را بفرست (برای رد شدن: 0):")
            return
        await _paper_finish_open(update, context, chat)
        return
    if data.startswith("p:close:"):
        pos_id = int(data.split(":", 2)[2])
        try:
            rec = await asyncio.to_thread(paper.close_position, cid, pos_id)
            src = "🤖 خودکار" if rec.get("source") == "auto" else "✋ دستی"
            await query.message.reply_text(
                f"✅ پوزیشن #{rec['id']} ({src}) بسته شد | خروج: {_fmt_price(rec['exit_price'])} | "
                f"سود/زیان خالص: {rec['net_pnl']:+,.2f} USDT",
                reply_markup=kb_paper(cid))
        except Exception as exc:
            await query.message.reply_text(f"❌ بستن پوزیشن ناموفق بود: {exc}",
                                           reply_markup=kb_paper(cid))
        return


async def paper_job(context: ContextTypes.DEFAULT_TYPE):
    """Periodic job (60 seconds): stop/target evaluation + auto tick for all accounts."""
    try:
        chat_ids = await asyncio.to_thread(paper.all_chat_ids)
    except Exception:
        return
    for cid in chat_ids:
        try:
            acct = paper.get_account(cid)
            if not acct["positions"] and not acct["auto"]["enabled"]:
                continue
            events = await asyncio.to_thread(paper.evaluate, cid)
            for ev in events:
                await context.bot.send_message(chat_id=int(cid), text=ev["message"])
            if acct["auto"]["enabled"]:
                actions = await asyncio.to_thread(paper.auto_tick, cid)
                for act in actions:
                    if "بسته شد" not in act:  # Closures were already announced by the evaluate call above
                        await context.bot.send_message(chat_id=int(cid), text=act)
        except Exception:
            log.exception("paper job failed for chat %s", cid)


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not _allowed(update):
        return
    await query.answer()
    data = query.data or ""
    chat = query.message.chat

    if data in ("m:home", "m:help"):
        await query.message.reply_text(HELP if data == "m:help" else WELCOME,
                                       parse_mode=ParseMode.HTML, reply_markup=kb_main())
        return
    if data.startswith("p:"):
        try:
            await on_paper_callback(update, context, data)
        except Exception as exc:
            log.exception("paper flow failed")
            await chat.send_message(f"❌ خطا در معاملات کاغذی: {exc}", reply_markup=kb_paper(chat.id))
        return
    if data == "m:analyze":
        context.user_data["await_symbol"] = True
        await query.message.reply_text("نماد ارز را بفرست (مثل BTC یا ETH):")
        return
    if data == "m:scan":
        status = await query.message.reply_text("🔍 در حال اسکن بازار...")
        try:
            await _run_scan_flow(chat, status)
        except Exception as exc:
            log.exception("scan failed")
            await chat.send_message(f"❌ خطا در اسکن: {exc}", reply_markup=kb_main())
        return
    if data == "m:fxscan":
        status = await query.message.reply_text("💱 در حال اسکن فارکس...")
        try:
            await _run_forex_scan_flow(chat, status)
        except Exception as exc:
            log.exception("forex scan failed")
            await chat.send_message(f"❌ خطا در اسکن فارکس: {exc}", reply_markup=kb_main())
        return
    if data.startswith("tf:"):
        _, symbol, timeframe = data.split(":", 2)
        status = await query.message.reply_text(f"⏳ در حال تحلیل {symbol} {timeframe} ...")
        try:
            df, report = await _do_analysis(chat, symbol, timeframe, status)
            context_cache.setdefault("last", {})[chat.id] = (symbol, timeframe, df, report)
        except Exception as exc:
            log.exception("analysis failed")
            try:
                await status.delete()
            except Exception:
                pass
            await chat.send_message(f"❌ خطا: {exc}", reply_markup=kb_main())
        return
    if data.startswith("ch:"):
        target = data[3:]
        entry = None
        if target == "top":
            entry = context_cache.get("top")
        else:
            symbol, timeframe = target.split(":", 1)
            last = context_cache.get("last", {}).get(chat.id)
            if last and last[0].upper() == symbol.upper() and last[1] == timeframe:
                entry = (last[2], last[3])
        if not entry:
            await query.message.reply_text("نمودار این مورد در دسترس نیست؛ تحلیل را دوباره اجرا کن.",
                                           reply_markup=kb_main())
            return
        df, report = entry
        try:
            png = await asyncio.to_thread(_chart_bytes, df, report)
            await chat.send_photo(photo=io.BytesIO(png),
                                  caption=f"🖼 {report['symbol'].upper()}/USDT · {report['timeframe']}")
        except Exception as exc:
            log.exception("chart failed")
            await chat.send_message(f"❌ خطا در رسم نمودار: {exc}")
        return


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update) or not update.message:
        return
    raw = (update.message.text or "").strip()
    if not raw:
        return
    # Paper-trading text flow (symbol → custom amount → manual stop loss/target)
    paper_await = context.user_data.get("paper_await")
    if paper_await:
        flow = context.user_data.setdefault("paper_flow", {})
        chat = update.message.chat
        if paper_await == "symbol":
            flow["symbol"] = raw.replace("/", " ").split()[0].upper()
            context.user_data.pop("paper_await", None)
            await update.message.reply_text(
                f"جهت پوزیشن {flow['symbol']} را انتخاب کن:",
                reply_markup=kb_paper_direction())
            return
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            await update.message.reply_text("عدد معتبر بفرست (مثل 150):")
            return
        if paper_await == "amount":
            flow["amount"] = value
            context.user_data.pop("paper_await", None)
            await update.message.reply_text("حد ضرر و هدف چطور باشد؟",
                                            reply_markup=kb_paper_mode())
            return
        if paper_await == "stop":
            flow["stop"] = value if value > 0 else None
            context.user_data["paper_await"] = "target"
            await update.message.reply_text("قیمت هدف را بفرست (برای رد شدن: 0):")
            return
        if paper_await == "target":
            flow["target"] = value if value > 0 else None
            context.user_data.pop("paper_await", None)
            flow["mode"] = "manual"
            await _paper_finish_open(update, context, chat)
            return
    parts = raw.replace("/", " ").split()
    symbol = parts[0].upper()
    timeframe = None
    if len(parts) > 1 and parts[1].lower() in TIMEFRAMES:
        timeframe = parts[1].lower()
    if timeframe:
        status = await update.message.reply_text(f"⏳ در حال تحلیل {symbol} {timeframe} ...")
        try:
            df, report = await _do_analysis(update.message.chat, symbol, timeframe, status)
            context_cache.setdefault("last", {})[update.message.chat.id] = (symbol, timeframe, df, report)
        except Exception as exc:
            log.exception("analysis failed")
            try:
                await status.delete()
            except Exception:
                pass
            await update.message.reply_text(f"❌ خطا: {exc}", reply_markup=kb_main())
        return
    if context.user_data.pop("await_symbol", False) or symbol.isalpha():
        await update.message.reply_text(
            f"تایم‌فریم {symbol} را انتخاب کن:", reply_markup=kb_timeframes(symbol))
        return


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    if app.job_queue is not None:
        app.job_queue.run_repeating(paper_job, interval=60, first=60,
                                    name="paper_trading_tick")
    log.info("bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
