"""Symbol catalog, normalisation and search.

The engine accepts user input such as ``BTC``, ``BTC/USDT``, ``btc-usd``,
``THYAO``, ``THYAO.IS`` or ``AAPL`` and resolves it to a canonical
(exchange ready) symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Curated catalog (used by the UI and for friendly names)
# ---------------------------------------------------------------------------

CRYPTO: List[Tuple[str, str]] = [
    ("BTCUSDT", "Bitcoin"),
    ("ETHUSDT", "Ethereum"),
    ("BNBUSDT", "BNB"),
    ("SOLUSDT", "Solana"),
    ("XRPUSDT", "XRP"),
    ("DOGEUSDT", "Dogecoin"),
    ("ADAUSDT", "Cardano"),
    ("AVAXUSDT", "Avalanche"),
    ("LINKUSDT", "Chainlink"),
    ("DOTUSDT", "Polkadot"),
    ("TRXUSDT", "TRON"),
    ("LTCUSDT", "Litecoin"),
    ("BCHUSDT", "Bitcoin Cash"),
    ("TONUSDT", "Toncoin"),
    ("SHIBUSDT", "Shiba Inu"),
    ("UNIUSDT", "Uniswap"),
    ("ATOMUSDT", "Cosmos"),
    ("ETCUSDT", "Ethereum Classic"),
    ("FILUSDT", "Filecoin"),
    ("APTUSDT", "Aptos"),
    ("ARBUSDT", "Arbitrum"),
    ("OPUSDT", "Optimism"),
    ("NEARUSDT", "NEAR Protocol"),
    ("INJUSDT", "Injective"),
    ("SUIUSDT", "Sui"),
    ("PEPEUSDT", "Pepe"),
    ("RNDRUSDT", "Render"),
    ("AAVEUSDT", "Aave"),
    ("MKRUSDT", "Maker"),
    ("SEIUSDT", "Sei"),
]

US: List[Tuple[str, str]] = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"),
    ("META", "Meta Platforms"),
    ("TSLA", "Tesla"),
    ("AVGO", "Broadcom"),
    ("AMD", "Advanced Micro Devices"),
    ("NFLX", "Netflix"),
    ("JPM", "JPMorgan Chase"),
    ("BAC", "Bank of America"),
    ("GS", "Goldman Sachs"),
    ("MS", "Morgan Stanley"),
    ("V", "Visa"),
    ("MA", "Mastercard"),
    ("WMT", "Walmart"),
    ("COST", "Costco"),
    ("JNJ", "Johnson & Johnson"),
    ("LLY", "Eli Lilly"),
    ("XOM", "Exxon Mobil"),
    ("CVX", "Chevron"),
    ("PLTR", "Palantir"),
    ("COIN", "Coinbase"),
    ("MSTR", "MicroStrategy"),
    ("SPY", "S&P 500 ETF"),
    ("QQQ", "Nasdaq 100 ETF"),
    ("IWM", "Russell 2000 ETF"),
    ("DIA", "Dow Jones ETF"),
    ("GLD", "Gold ETF"),
]

BIST: List[Tuple[str, str]] = [
    ("THYAO", "Türk Hava Yolları"),
    ("GARAN", "Garanti BBVA"),
    ("ASELS", "Aselsan"),
    ("EREGL", "Ereğli Demir Çelik"),
    ("BIMAS", "BİM Birleşik Mağazalar"),
    ("AKBNK", "Akbank"),
    ("SISE", "Şişecam"),
    ("KOZAL", "Koza Altın"),
    ("SAHOL", "Sabancı Holding"),
    ("TUPRS", "Tüpraş"),
    ("KCHOL", "Koç Holding"),
    ("HALKB", "Halkbank"),
    ("VAKBN", "Vakıfbank"),
    ("TCELL", "Turkcell"),
    ("PGSUS", "Pegasus"),
    ("PETKM", "Petkim"),
    ("TOASO", "Tofaş"),
    ("FROTO", "Ford Otosan"),
    ("DOHOL", "Doğan Holding"),
    ("TAVHL", "TAV Havalimanları"),
    ("SASA", "Sasa Polyester"),
    ("HEKTS", "Hektaş"),
    ("KRDMD", "Kardemir"),
    ("ISCTR", "İş Bankası C"),
    ("YKBNK", "Yapı Kredi"),
    ("TTKOM", "Türk Telekom"),
    ("ENKAI", "Enka İnşaat"),
    ("MGROS", "Migros"),
    ("ULKER", "Ülker Bisküvi"),
    ("ARCLK", "Arçelik"),
]

_CRYPTO_NAMES: Dict[str, str] = dict(CRYPTO)
_US_NAMES: Dict[str, str] = dict(US)
_BIST_NAMES: Dict[str, str] = dict(BIST)

CRYPTO_BASES = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "TRX", "LTC", "BCH", "TON", "SHIB", "UNI", "ATOM", "ETC", "FIL", "APT",
    "ARB", "OP", "NEAR", "INJ", "SUI", "PEPE", "RNDR", "AAVE", "MKR", "SEI",
    "MATIC", "POL", "ICP", "XLM", "HBAR", "VET", "ALGO", "SAND", "MANA",
    "AXS", "GALA", "FTM", "RUNE", "CRV", "LDO", "WLD", "JUP", "TIA", "ORDI",
    "1000PEPE", "1000SHIB", "WIF", "BONK", "FLOKI", "NOT", "ENA", "ETHFI",
}

_ALIASES = {
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    "SOLANA": "SOL",
    "DOGECOIN": "DOGE",
    "CARDANO": "ADA",
    "POLKADOT": "DOT",
    "CHAINLINK": "LINK",
    "AVALANCHE": "AVAX",
    "LITECOIN": "LTC",
    "RIPPLE": "XRP",
    "BINANCECOIN": "BNB",
    "SHIBAINU": "SHIB",
    "TONCOIN": "TON",
    "TRON": "TRX",
}

_QUOTES = ("USDT", "USDC", "FDUSD", "BUSD", "TUSD", "TRY", "BTC", "ETH", "EUR")

MARKET_LABELS = {
    "crypto": "Kripto",
    "us": "ABD Borsası",
    "bist": "Borsa İstanbul",
}

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


@dataclass(frozen=True)
class ResolvedSymbol:
    """A user supplied symbol resolved to a canonical market symbol."""

    user_input: str
    symbol: str          # display symbol, e.g. BTCUSDT / THYAO / AAPL
    data_symbol: str     # provider symbol, e.g. BTCUSDT / THYAO.IS / AAPL
    market: str          # crypto / us / bist
    currency: str
    name: str
    base: str = ""
    quote: str = ""

    @property
    def market_label(self) -> str:
        return MARKET_LABELS.get(self.market, self.market)


def _fold(value: str) -> str:
    """Case-insensitive folding that also handles Turkish dotted/dotless i."""
    return (
        value.replace("İ", "i")
        .replace("I", "ı")
        .casefold()
        .strip()
    )


def _clean(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def _split_crypto(pair: str) -> Tuple[str, str]:
    """Split a concatenated crypto pair into base and quote."""
    for quote in sorted(_QUOTES, key=len, reverse=True):
        if pair.endswith(quote) and len(pair) > len(quote):
            return pair[: -len(quote)], quote
    return pair, ""


def looks_like_crypto(raw: str) -> bool:
    """Heuristic used when ``market='auto'``."""
    value = _clean(raw)
    if not value:
        return False
    if "/" in value or "-" in value:
        parts = value.replace("-", "/").split("/")
        return parts[0] in CRYPTO_BASES or parts[-1] in _QUOTES
    if value in _ALIASES or value in CRYPTO_BASES:
        return True
    base, quote = _split_crypto(value)
    return bool(quote) and (base in CRYPTO_BASES or len(base) <= 10)


def _crypto_name(base: str) -> str:
    for quote in _QUOTES:
        name = _CRYPTO_NAMES.get(f"{base}{quote}")
        if name:
            return name
    return base.title()


def _resolve_crypto(raw: str) -> ResolvedSymbol:
    value = _clean(raw).replace("/", "").replace("-", "").replace("_", "")
    if value in _ALIASES:
        value = _ALIASES[value]
    if value.endswith("USD") and not value.endswith("USDT"):
        value = value[:-3] + "USDT"

    base, quote = _split_crypto(value)
    if not quote:
        base = value
        quote = "USDT"
        value = f"{base}{quote}"

    if not base:
        raise ValueError(f"Geçersiz kripto sembolü: {raw}")

    # If the user typed a stablecoin quote that Binance does not list as spot
    # (e.g. BUSD), fall back to USDT.
    if quote == "BUSD":
        quote = "USDT"
        value = f"{base}{quote}"

    return ResolvedSymbol(
        user_input=raw,
        symbol=value,
        data_symbol=value,
        market="crypto",
        currency=quote if quote not in ("USDT", "USDC", "FDUSD", "BUSD") else "USD",
        name=_crypto_name(base),
        base=base,
        quote=quote,
    )


def _resolve_bist(raw: str) -> ResolvedSymbol:
    display = _clean(raw)
    if display.endswith(".IS"):
        display = display[:-3]
    if not display:
        raise ValueError("Boş BIST sembolü")
    return ResolvedSymbol(
        user_input=raw,
        symbol=display,
        data_symbol=f"{display}.IS",
        market="bist",
        currency="TRY",
        name=_BIST_NAMES.get(display, display),
        base=display,
        quote="TRY",
    )


def _resolve_us(raw: str) -> ResolvedSymbol:
    display = _clean(raw)
    if not display:
        raise ValueError("Boş sembol")
    return ResolvedSymbol(
        user_input=raw,
        symbol=display,
        data_symbol=display,
        market="us",
        currency="USD",
        name=_US_NAMES.get(display, display),
        base=display,
        quote="USD",
    )


def resolve_symbol(raw: str, market: str = "auto") -> ResolvedSymbol:
    """Resolve user input into a :class:`ResolvedSymbol`.

    Parameters
    ----------
    raw:
        Symbol as typed by the user. Examples: ``BTC``, ``BTCUSDT``,
        ``ETH/USDT``, ``THYAO``, ``THYAO.IS``, ``AAPL``.
    market:
        ``auto`` (default), ``crypto``, ``us`` or ``bist``/``turkey``.
    """
    if not raw or not str(raw).strip():
        raise ValueError("Sembol boş olamaz")

    market = (market or "auto").strip().lower()
    if market in ("bist", "borsa", "istanbul", "turkey", "tr"):
        market = "bist"
    elif market in ("stock", "stocks", "nasdaq", "nyse"):
        market = "us"

    if market == "crypto":
        return _resolve_crypto(raw)
    if market == "bist":
        return _resolve_bist(raw)
    if market == "us":
        return _resolve_us(raw)

    # auto
    value = _clean(raw)
    if value.endswith(".IS"):
        return _resolve_bist(raw)
    if value in _BIST_NAMES:
        return _resolve_bist(raw)
    if looks_like_crypto(raw):
        return _resolve_crypto(raw)
    return _resolve_us(raw)


def catalog() -> Dict[str, List[dict]]:
    """Return the full curated catalog grouped by market."""
    return {
        "crypto": [
            {"symbol": symbol, "name": name, "market": "crypto",
             "currency": "USD", "label": f"{symbol[:-4] if symbol.endswith('USDT') else symbol} · {name}"}
            for symbol, name in CRYPTO
        ],
        "us": [
            {"symbol": symbol, "name": name, "market": "us",
             "currency": "USD", "label": f"{symbol} · {name}"}
            for symbol, name in US
        ],
        "bist": [
            {"symbol": symbol, "name": name, "market": "bist",
             "currency": "TRY", "label": f"{symbol} · {name}"}
            for symbol, name in BIST
        ],
    }


def search_symbols(query: str = "", market: str = "auto", limit: int = 15) -> List[dict]:
    """Search the curated catalog by symbol or company name."""
    query_folded = _fold(query or "")
    market = (market or "auto").lower()
    if market == "turkey":
        market = "bist"

    results: List[dict] = []
    for market_key, items in catalog().items():
        if market not in ("auto", "", market_key):
            continue
        for item in items:
            haystack = _fold(f"{item['symbol']} {item['name']}")
            if not query_folded or query_folded in haystack:
                score = 0
                if query_folded == _fold(item["symbol"]):
                    score = 100
                elif _fold(item["symbol"]).startswith(query_folded):
                    score = 80
                elif query_folded in _fold(item["symbol"]):
                    score = 60
                elif query_folded in _fold(item["name"]):
                    score = 40
                results.append({**item, "score": score})

    results.sort(key=lambda row: (-row["score"], row["symbol"]))
    return results[: max(1, int(limit))]


def interval_ms(interval: str) -> int:
    try:
        return _INTERVAL_MS[interval]
    except KeyError as exc:
        raise ValueError(
            f"Desteklenmeyen zaman aralığı: {interval}. "
            f"Geçerli: {', '.join(_INTERVAL_MS)}"
        ) from exc


def supported_intervals(market: str = "auto") -> List[str]:
    if market == "crypto":
        return ["1h", "4h", "1d", "1w"]
    return ["1h", "1d", "1w"]


def popular_symbols() -> Dict[str, List[str]]:
    return {
        "crypto": [s for s, _ in CRYPTO[:12]],
        "us": [s for s, _ in US[:12]],
        "bist": [s for s, _ in BIST[:12]],
    }
