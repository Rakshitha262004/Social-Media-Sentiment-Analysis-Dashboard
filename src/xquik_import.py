#!/usr/bin/env python3
# ============================================================
# File: src/xquik_import.py
# Purpose: Convert Xquik tweet exports into dashboard CSV rows.
# ============================================================

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("data") / "xquik_posts.csv"
OUTPUT_COLUMNS = ["id", "platform", "brand", "text", "sentiment", "likes", "retweets"]


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return [row for row in rows if isinstance(row, dict)]

    if isinstance(payload, dict):
        tweets = payload.get("tweets", [])
        if isinstance(tweets, list):
            return [tweet for tweet in tweets if isinstance(tweet, dict)]
        return [payload]
    if isinstance(payload, list):
        return [tweet for tweet in payload if isinstance(tweet, dict)]
    return []


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    return _load_json_or_jsonl(path)


def author_name(tweet: dict[str, Any]) -> str:
    author = tweet.get("author")
    if isinstance(author, dict):
        value = author.get("username") or author.get("userName") or author.get("name")
        if value:
            return str(value)
    value = tweet.get("username") or tweet.get("userName") or tweet.get("author")
    return str(value or "Xquik")


def metric_value(tweet: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = tweet.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    brand: str,
    default_sentiment: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, tweet in enumerate(rows, start=1):
        text = str(tweet.get("text") or tweet.get("content") or "").strip()
        if not text:
            continue

        row_id = str(tweet.get("id") or tweet.get("tweet_id") or index)
        dedupe_key = row_id if row_id else text
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        records.append(
            {
                "id": row_id,
                "platform": "Twitter",
                "brand": brand or author_name(tweet),
                "text": text,
                "sentiment": str(tweet.get("sentiment") or default_sentiment).lower(),
                "likes": metric_value(tweet, "likeCount", "like_count", "likes"),
                "retweets": metric_value(tweet, "retweetCount", "retweet_count", "retweets"),
            }
        )

    return records


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    brand: str,
    default_sentiment: str,
) -> list[dict[str, Any]]:
    rows = load_rows(input_path)
    records = normalize_rows(rows, brand=brand, default_sentiment=default_sentiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Xquik tweet JSON, JSONL, or CSV exports for the dashboard."
    )
    parser.add_argument("input", type=Path, help="Xquik tweet export file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path, defaults to {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--brand", default="Xquik", help="Brand label for imported rows")
    parser.add_argument(
        "--sentiment",
        default="neutral",
        choices=["positive", "negative", "neutral"],
        help="Default sentiment for unlabeled Xquik rows",
    )
    args = parser.parse_args()

    records = convert_file(
        args.input,
        args.output,
        brand=args.brand,
        default_sentiment=args.sentiment,
    )
    print(f"Saved {len(records)} Xquik rows to {args.output}")


if __name__ == "__main__":
    main()
