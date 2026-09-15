from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import StrategyConfig
from .models import HedgeSnapshot, parse_utc_datetime
from .service import FundingHedgeService


UTC = timezone.utc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily hedge a paying Hyperliquid position in an independent "
            "subaccount/wallet without modifying the primary position."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Evaluate offline JSON instead of reading Hyperliquid.",
    )
    parser.add_argument(
        "--now",
        help="Override evaluation time with an ISO-8601 timestamp (offline only).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling; default is one evaluation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Allow hedge-account orders. Also requires "
            "FUNDING_HEDGE_EXECUTE=true; never routes an order to the primary."
        ),
    )
    return parser


def _read_snapshot(path: Path) -> List[HedgeSnapshot]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_snapshots = payload.get("snapshots") if isinstance(payload, dict) else payload
    if not isinstance(raw_snapshots, list):
        raise ValueError("snapshot must be a list or an object containing snapshots")
    return [HedgeSnapshot.from_mapping(item) for item in raw_snapshots]


def _render(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.now and args.snapshot is None:
        parser.error("--now requires --snapshot")
    if args.now and args.watch:
        parser.error("--now cannot be combined with --watch")
    if args.execute and args.snapshot is not None:
        parser.error("--execute cannot be combined with --snapshot")
    if args.execute and not args.watch:
        parser.error("--execute requires --watch so an opened hedge is managed to flat")
    try:
        config = StrategyConfig.from_env()
        service = FundingHedgeService(config) if args.snapshot is None else None
        fixed_now = parse_utc_datetime(args.now, "now") if args.now else None
        while True:
            now = fixed_now or datetime.now(tz=UTC)
            if args.snapshot is not None:
                snapshots = _read_snapshot(args.snapshot)
                output = FundingHedgeService.evaluate_snapshots(
                    config, snapshots, now
                )
            else:
                assert service is not None
                output = service.run_once(now, execute=args.execute)
            _render(output)
            if not args.watch:
                return 0
            time.sleep(config.poll_seconds)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"funding_fee_avoidance: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
