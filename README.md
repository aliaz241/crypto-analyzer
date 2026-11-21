# Crypto Analyzer

A Python technical-analysis tool for the crypto (and forex) markets:
indicators, candlestick and chart patterns, backtesting, EV calculation,
and a multi-symbol scanner + Telegram bot.

## Setup

```bash
pip install -r requirements.txt
```

You must supply the keys and token **yourself** (in the file `crypto_analyzer/config.py`):

- `BOT_TOKEN` in `crypto_analyzer/telegram_bot.py` — get your Telegram bot token from
  [@BotFather](https://t.me/BotFather) and replace it.
- `COINGECKO_API_KEY` — optional, your own CoinGecko key.
- `NVIDIA_API_KEY` — optional, only for the LLM layer.

## Running

```bash
python -m crypto_analyzer.scanner
python -m crypto_analyzer.telegram_bot
```

> This tool is for analysis and education only and is not financial advice.
