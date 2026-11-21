# Crypto Analyzer

ابزار تحلیل تکنیکال بازار کریپتو (و فارکس) با پایتون: اندیکاتورها، الگوهای
کندلی و چارتی، بک‌تست، محاسبه‌ی EV و اسکنر چندنمادی + ربات تلگرام.

## راه‌اندازی

```bash
pip install -r requirements.txt
```

کلیدها و توکن را باید **خودت** بگذاری (در فایل `crypto_analyzer/config.py`):

- `BOT_TOKEN` در `crypto_analyzer/telegram_bot.py` — توکن ربات تلگرامت را از
  [@BotFather](https://t.me/BotFather) بگیر و جایگزین کن.
- `COINGECKO_API_KEY` — اختیاری، کلید CoinGecko خودت.
- `NVIDIA_API_KEY` — اختیاری، فقط برای لایه‌ی LLM.

## اجرا

```bash
python -m crypto_analyzer.scanner
python -m crypto_analyzer.telegram_bot
```

> این ابزار صرفاً برای تحلیل و آموزش است و توصیه‌ی مالی نیست.
