"""
Build catalog.json from SHL's hosted product-catalog JSON feed.

This replaces the original plan of scraping shl.com's paginated HTML
listing/detail pages (see approach.md's superseded "Data" section) now that
a single pre-scraped JSON feed is available at CATALOG_URL. It's the same
underlying data (SHL's Individual Test Solutions + Pre-packaged Job
Solutions catalog), already flattened to one record per assessment, so
there's no HTML to parse and no pagination to walk -- just fetch, validate,
and reshape into the schema CatalogEntry (retrieval.py) expects.

Run:
    python fetch_catalog.py [-o catalog.json] [--include-failed]

Source record shape (one dict per assessment), as observed on the feed:
    {
      "entity_id": "4302",
      "name": "Global Skills Development Report",
      "link": "https://www.shl.com/products/product-catalog/view/.../",
      "scraped_at": "2026-05-08T10:40:21.464836+00:00",
      "job_levels": ["Director", "Entry-Level", ...],
      "job_levels_raw": "Director, Entry-Level, ...,",
      "languages": ["English (USA)"],
      "languages_raw": "English (USA),",
      "duration": "30 minutes" | "" | "Variable" | "Untimed" | "0 minutes",
      "duration_raw": "Approximate Completion Time in minutes = 30",
      "status": "ok" | other,
      "remote": "yes" | "no",
      "adaptive": "yes" | "no",
      "description": "...",
      "keys": ["Knowledge & Skills", "Simulations", ...]   # test-type labels
    }

Output record shape (one dict per assessment) -- matches CatalogEntry in
retrieval.py / the fields agent.py hands back to the model:
    {
      "name": str,
      "url": str,
      "description": str,
      "job_levels": [str, ...],
      "languages": [str, ...],
      "duration_minutes": int | None,      # None for Variable/Untimed/blank
      "test_type": ["K", "S", ...],        # letter codes, see TEST_TYPE_CODES
      "test_type_labels": [str, ...],      # original "keys" labels, kept
      "remote_testing": bool | None,
      "adaptive_irt": bool | None,
    }

Known caveat, carried over from the original approach.md: the feed has no
field distinguishing "Individual Test Solutions" from "Pre-packaged Job
Solutions" (the type=1 vs type=2 split on shl.com's own catalog page, which
the assignment says to exclude). Names like "... Solution" (e.g. "Entry
Level Cashier Solution") strongly suggest packaged solutions are present in
this feed. EXCLUDE_NAME_HINTS below does a conservative, inspectable
best-effort filter on that naming pattern; it is NOT a substitute for
checking against the live site's type=2 listing before treating catalog.json
as authoritative. Flagging this rather than silently guessing.
"""
import argparse
import json
import re
import sys
import urllib.request
from urllib.error import URLError, HTTPError

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

# "keys" (free-text test-type labels) -> single-letter codes, per the
# legend in agent.py's search_catalog tool description.
TEST_TYPE_CODES = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Conservative, inspectable heuristic for "Pre-packaged Job Solutions" --
# see the caveat above. Verify against shl.com?type=2 before trusting this.
EXCLUDE_NAME_HINTS = re.compile(r"\bsolution\b", re.IGNORECASE)

_DURATION_NUM_RE = re.compile(r"\d+")


def _parse_duration_minutes(duration: str) -> int | None:
    """'30 minutes' -> 30; '' / 'Variable' / 'Untimed' -> None."""
    if not duration:
        return None
    m = _DURATION_NUM_RE.search(duration)
    if not m:
        return None
    minutes = int(m.group())
    return minutes if minutes > 0 else None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def transform(raw_records: list[dict], *, include_failed: bool, exclude_solutions: bool) -> list[dict]:
    out = []
    skipped_status = 0
    skipped_solution = 0
    for r in raw_records:
        if not include_failed and r.get("status") != "ok":
            skipped_status += 1
            continue
        name = (r.get("name") or "").strip()
        url = (r.get("link") or "").strip()
        if not name or not url:
            continue
        if exclude_solutions and EXCLUDE_NAME_HINTS.search(name):
            skipped_solution += 1
            continue

        labels = r.get("keys") or []
        codes = sorted({TEST_TYPE_CODES[label] for label in labels if label in TEST_TYPE_CODES})
        unmapped = [label for label in labels if label not in TEST_TYPE_CODES]
        if unmapped:
            print(f"warning: unmapped test-type label(s) {unmapped!r} on {name!r}", file=sys.stderr)

        out.append({
            "name": name,
            "url": url,
            "description": (r.get("description") or "").strip(),
            "job_levels": r.get("job_levels") or [],
            "languages": r.get("languages") or [],
            "duration_minutes": _parse_duration_minutes(r.get("duration") or ""),
            "test_type": codes,
            "test_type_labels": labels,
            "remote_testing": _parse_bool(r.get("remote")),
            "adaptive_irt": _parse_bool(r.get("adaptive")),
        })

    if skipped_status:
        print(f"skipped {skipped_status} record(s) with status != 'ok' "
              f"(pass --include-failed to keep them)", file=sys.stderr)
    if skipped_solution:
        print(f"skipped {skipped_solution} record(s) matching the "
              f"'Pre-packaged Job Solutions' name heuristic "
              f"(pass --keep-solutions to keep them)", file=sys.stderr)
    return out


def fetch(url: str = CATALOG_URL, timeout: int = 60) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "shl-catalog-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (URLError, HTTPError) as e:
        raise SystemExit(f"failed to fetch catalog feed from {url}: {e}")
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise SystemExit(f"catalog feed at {url} was not valid JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default="catalog.json", help="output path (default: catalog.json)")
    parser.add_argument("--url", default=CATALOG_URL, help="override the source feed URL")
    parser.add_argument("--include-failed", action="store_true",
                         help="keep records whose status != 'ok' instead of dropping them")
    parser.add_argument("--keep-solutions", action="store_true",
                         help="keep records matching the packaged-solution name heuristic")
    args = parser.parse_args()

    raw = fetch(args.url)
    if not isinstance(raw, list):
        raise SystemExit(f"expected a JSON array of records, got {type(raw).__name__}")

    records = transform(
        raw,
        include_failed=args.include_failed,
        exclude_solutions=not args.keep_solutions,
    )
    if not records:
        raise SystemExit("transform produced zero usable records -- check the feed / filters")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(records)} records to {args.output} "
          f"(from {len(raw)} source records)")


if __name__ == "__main__":
    main()
