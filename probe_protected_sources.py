from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable

from dotenv import load_dotenv

from watchers.bls_watcher import build_bls_watchers
from watchers.bloomberg_watcher import build_bloomberg_watchers
from watchers.dol_watcher import build_dol_watchers
from watchers.irna_watcher import build_irna_watchers
from watchers.truth_social_watcher import build_truth_social_watchers


load_dotenv()


@dataclass
class ProbeResult:
    source: str
    target: str
    enabled: bool
    has_cookie: bool
    ok: bool
    item_count: int = 0
    sample_title: str = ""
    error: str = ""


@dataclass
class ProbeSpec:
    name: str
    builder: Callable[[], list[Any]]
    cookie_env: str
    target_attr: str
    probe_fn: Callable[[Any], tuple[int, str]]


def probe_html_watcher(watcher: Any) -> tuple[int, str]:
    html = watcher._fetch_html()
    items = watcher.extract_events(html)
    sample_title = items[0].title if items else ""
    return len(items), sample_title


def probe_rss_watcher(watcher: Any) -> tuple[int, str]:
    items = watcher._fetch_events()
    sample_title = items[0].title if items else ""
    return len(items), sample_title


def probe_event_watcher(watcher: Any) -> tuple[int, str]:
    items = watcher._fetch_events()
    sample_title = items[0].title if items else ""
    return len(items), sample_title


def probe_truth_social_watcher(watcher: Any) -> tuple[int, str]:
    posts = watcher._safe_fetch_statuses()
    originals = [post for post in posts if not watcher._is_repost(post)]
    sample_title = watcher._summarize_post_line(originals[-1]) if originals else ""
    return len(originals), sample_title


PROBE_SPECS: dict[str, ProbeSpec] = {
    "truth_social": ProbeSpec(
        name="truth_social",
        builder=build_truth_social_watchers,
        cookie_env="TRUTHSOCIAL_COOKIE_HEADER",
        target_attr="handle",
        probe_fn=probe_truth_social_watcher,
    ),
    "irna": ProbeSpec(
        name="irna",
        builder=build_irna_watchers,
        cookie_env="IRNA_COOKIE_HEADER",
        target_attr="url",
        probe_fn=probe_html_watcher,
    ),
    "bloomberg": ProbeSpec(
        name="bloomberg",
        builder=build_bloomberg_watchers,
        cookie_env="BLOOMBERG_COOKIE_HEADER",
        target_attr="target_url",
        probe_fn=probe_event_watcher,
    ),
    "dol": ProbeSpec(
        name="dol",
        builder=build_dol_watchers,
        cookie_env="DOL_COOKIE_HEADER",
        target_attr="url",
        probe_fn=probe_html_watcher,
    ),
    "bls": ProbeSpec(
        name="bls",
        builder=build_bls_watchers,
        cookie_env="BLS_COOKIE_HEADER",
        target_attr="feed_url",
        probe_fn=probe_rss_watcher,
    ),
}


def _watcher_has_cookie(watcher: Any) -> bool:
    session = getattr(watcher, "session", None)
    if session is not None:
        return bool((session.headers.get("Cookie") or "").strip())
    client = getattr(watcher, "client", None)
    client_session = getattr(client, "session", None)
    if client_session is not None:
        return bool((client_session.headers.get("Cookie") or "").strip())
    return False


def probe_source(spec: ProbeSpec) -> list[ProbeResult]:
    watchers = spec.builder()
    if not watchers:
        return [
            ProbeResult(
                source=spec.name,
                target="",
                enabled=False,
                has_cookie=False,
                ok=False,
                error="not enabled",
            )
        ]

    results: list[ProbeResult] = []
    for watcher in watchers:
        target = getattr(watcher, spec.target_attr, "")
        if spec.name == "truth_social" and target:
            target = f"@{target}"
        try:
            item_count, sample_title = spec.probe_fn(watcher)
            results.append(
                ProbeResult(
                    source=watcher.source_name,
                    target=target,
                    enabled=True,
                    has_cookie=_watcher_has_cookie(watcher),
                    ok=True,
                    item_count=item_count,
                    sample_title=sample_title,
                )
            )
        except Exception as e:
            results.append(
                ProbeResult(
                    source=watcher.source_name,
                    target=target,
                    enabled=True,
                    has_cookie=_watcher_has_cookie(watcher),
                    ok=False,
                    error=str(e),
                )
            )
    return results


def run_selected_probes(names: list[str]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for name in names:
        spec = PROBE_SPECS[name]
        results.extend(probe_source(spec))
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Truth Social / IRNA / Bloomberg / DOL / BLS with current cookie headers")
    parser.add_argument(
        "--sources",
        default="truth_social,irna,bloomberg,dol,bls",
        help="Comma-separated subset of: truth_social,irna,bloomberg,dol,bls",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    names = [name.strip().lower() for name in args.sources.split(",") if name.strip()]
    unknown = [name for name in names if name not in PROBE_SPECS]
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(unknown)}")

    results = run_selected_probes(names)
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "OK" if result.ok else "FAIL"
            if not result.enabled:
                status = "SKIP"
            print(f"[{status}] {result.source}")
            print(f"  target     : {result.target or '-'}")
            print(f"  has_cookie : {result.has_cookie}")
            if result.ok:
                print(f"  item_count : {result.item_count}")
                print(f"  sample     : {result.sample_title or '-'}")
            else:
                print(f"  error      : {result.error}")
    return 0 if all((not result.enabled) or result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
