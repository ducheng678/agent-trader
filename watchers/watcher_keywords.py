import os
from dataclasses import dataclass
from typing import Iterable, Optional


TRADE_SYMBOL_TOPIC_KEYWORDS = {
    "BTC": [
        "bitcoin",
        "btc",
        "crypto",
        "digital asset",
        "stablecoin",
        "token",
        "etf",
        "exchange",
        "coinfund",
        "franklin templeton",
        "regulation",
        "sec",
        "treasury",
        "dxy",
        "dollar index",
        "rates",
        "fed",
        "liquidity",
    ],
    "ETH": [
        "ethereum",
        "eth",
        "crypto",
        "token",
        "stablecoin",
        "staking",
        "etf",
        "regulation",
        "sec",
        "treasury",
        "dxy",
        "dollar index",
        "rates",
        "fed",
    ],
    "SILVER": [
        "silver",
        "xag",
        "metal",
        "metals",
        "bullion",
        "precious",
        "mining",
        "industrial demand",
        "solar",
        "tariff",
        "dxy",
        "dollar index",
        "rates",
        "fed",
        "inflation",
    ],
    "BRENTOIL": [
        "brent",
        "oil",
        "crude",
        "opec",
        "shipping",
        "hormuz",
        "iran",
        "blockade",
        "naval blockade",
        "sanctions",
        "ceasefire",
        "middle east",
        "gulf",
        "refining",
        "gasoline",
        "diesel",
        "inventory",
        "inventories",
    ],
}


SUPPLEMENTAL_MACRO_KEYWORDS = [
    "dxy",
    "dollar index",
    "u.s. dollar index",
    "us dollar index",
    "treasury",
    "treasuries",
    "treasury yield",
    "treasury yields",
    "bond yield",
    "bond yields",
    "note yield",
    "note yields",
    "auction",
    "auctions",
    "adp",
    "jobs",
    "job growth",
    "payroll",
    "retail sales",
    "consumer spending",
    "census bureau",
]


@dataclass
class TradeTopicProfile:
    strong_keywords: list[str]
    support_keywords: list[str]


def _raw_trade_symbol_values(raw_values: Optional[Iterable[str]] = None) -> list[str]:
    if raw_values is not None:
        return [str(item or "") for item in raw_values]
    return [os.getenv("TRADE_SYMBOLS", ""), os.getenv("TRADE_SYMBOL", "")]


def dedupe_keywords(keywords: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = str(keyword or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def keywords_from_trade_symbols(raw_values: Optional[Iterable[str]] = None) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for raw in _raw_trade_symbol_values(raw_values):
        raw_upper = raw.upper()
        if not raw_upper:
            continue
        for key, additions in TRADE_SYMBOL_TOPIC_KEYWORDS.items():
            if key not in raw_upper:
                continue
            for keyword in additions:
                normalized = keyword.lower()
                if normalized not in seen:
                    seen.add(normalized)
                    keywords.append(normalized)
    for keyword in SUPPLEMENTAL_MACRO_KEYWORDS:
        normalized = keyword.lower()
        if normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)
    return keywords


def trade_topic_profiles(raw_values: Optional[Iterable[str]] = None) -> list[TradeTopicProfile]:
    profiles: list[TradeTopicProfile] = []
    seen_keys: set[str] = set()
    for raw in _raw_trade_symbol_values(raw_values):
        raw_upper = raw.upper()
        if not raw_upper:
            continue
        if "BTC" in raw_upper and "BTC" not in seen_keys:
            seen_keys.add("BTC")
            profiles.append(
                TradeTopicProfile(
                    strong_keywords=["bitcoin", "btc", "crypto", "stablecoin", "token", "etf", "exchange"],
                    support_keywords=["regulation", "sec", "treasury", "dxy", "dollar index", "rates", "fed", "liquidity"],
                )
            )
        if "ETH" in raw_upper and "ETH" not in seen_keys:
            seen_keys.add("ETH")
            profiles.append(
                TradeTopicProfile(
                    strong_keywords=["ethereum", "eth", "crypto", "token", "stablecoin", "staking", "etf"],
                    support_keywords=["regulation", "sec", "treasury", "dxy", "dollar index", "rates", "fed"],
                )
            )
        if ("SILVER" in raw_upper or "XAG" in raw_upper) and "SILVER" not in seen_keys:
            seen_keys.add("SILVER")
            profiles.append(
                TradeTopicProfile(
                    strong_keywords=["silver", "xag", "metal", "metals", "bullion", "mining", "industrial demand", "solar"],
                    support_keywords=["dxy", "dollar index", "rates", "fed", "inflation", "tariff"],
                )
            )
        if ("BRENTOIL" in raw_upper or "BRENT" in raw_upper) and "BRENTOIL" not in seen_keys:
            seen_keys.add("BRENTOIL")
            profiles.append(
                TradeTopicProfile(
                    strong_keywords=[
                        "brent",
                        "oil",
                        "crude",
                        "opec",
                        "shipping",
                        "hormuz",
                        "iran",
                        "blockade",
                        "naval blockade",
                        "refining",
                        "gasoline",
                        "diesel",
                        "inventory",
                        "inventories",
                    ],
                    support_keywords=["iran", "sanctions", "ceasefire", "middle east", "gulf"],
                )
            )
    profiles.append(
        TradeTopicProfile(
            strong_keywords=[
                "dxy",
                "dollar index",
                "u.s. dollar index",
                "us dollar index",
                "treasury",
                "treasuries",
                "treasury yield",
                "treasury yields",
                "bond yield",
                "bond yields",
                "note yield",
                "note yields",
            ],
            support_keywords=[
                "fed",
                "rates",
                "inflation",
                "job",
                "jobs",
                "payroll",
                "adp",
                "retail sales",
                "census",
                "auction",
                "auctions",
            ],
        )
    )
    return profiles


def profile_matches(profile: TradeTopicProfile, haystack: str) -> bool:
    if any(keyword in haystack for keyword in profile.strong_keywords):
        return True
    support_hits = sum(1 for keyword in profile.support_keywords if keyword in haystack)
    return support_hits >= 2
