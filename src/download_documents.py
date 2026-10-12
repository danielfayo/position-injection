"""Download and prepare the carrier documents (step 3 of the implementation plan).

Samples random English Wikipedia articles, keeps only their prose, and cuts each
one at a sentence boundary to about the target length in tokens. Writes one text
file per document and a manifest recording where each came from, which the
CC BY-SA licence requires when the documents are released.

The manifest is saved after every document, so if the run stops, rerunning the
same command carries on from where it stopped.

Usage:
    uv run python -m src.download_documents              # 30 documents into data/documents
    uv run python -m src.download_documents --overwrite  # start again from scratch
    uv run python -m src.download_documents --n 2 --output-dir /tmp/docs   # quick test
"""

import argparse
import csv
import datetime as dt
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.text import clean_extract, split_sentences

ROOT = Path(__file__).resolve().parent.parent
API = "https://en.wikipedia.org/w/api.php"
LICENCE = "CC BY-SA 4.0"
MANIFEST_FIELDS = [
    "doc_id", "title", "pageid", "revid", "revision_timestamp", "url",
    "licence", "n_tokens", "n_sentences", "retrieved_at",
]
REQUEST_DELAY = 1.0   # seconds between API calls
MAX_ATTEMPTS = 8


def make_session() -> requests.Session:
    # Wikimedia asks API clients to identify themselves; WIKI_CONTACT in .env is optional.
    load_dotenv(ROOT / ".env")
    contact = os.getenv("WIKI_CONTACT")
    session = requests.Session()
    session.headers["User-Agent"] = (
        "position-injection-research/0.1 (academic study of prompt injection"
        + (f"; contact: {contact})" if contact else ")")
    )
    return session


def api_query(session: requests.Session, **params) -> dict:
    """Call the MediaWiki query API, pausing between calls and backing off when refused."""
    params |= {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5}
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(REQUEST_DELAY)
        response = session.get(API, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("error", {}).get("code") == "maxlag":
                time.sleep(5)
                continue
            if "error" in data:
                raise RuntimeError(f"Wikipedia API error: {data['error']}")
            return data
        if response.status_code == 429 or response.status_code >= 500:
            # Wait as long as Wikipedia asks, or back off up to two minutes.
            retry_after = response.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else min(120, 5 * 2 ** attempt)
            print(f"   Wikipedia is busy (HTTP {response.status_code}); waiting {wait} s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
    raise RuntimeError("Wikipedia kept refusing requests. Wait a few minutes and rerun; it resumes.")


def random_candidates(session: requests.Session, min_bytes: int):
    """Yield random articles long enough to be worth fetching, skipping lists and disambiguation pages."""
    while True:
        data = api_query(
            session, generator="random", grnnamespace=0, grnlimit=500,
            prop="info|pageprops", ppprop="disambiguation",
        )
        for page in data["query"]["pages"]:
            if page.get("length", 0) < min_bytes:
                continue
            if "disambiguation" in page.get("pageprops", {}):
                continue
            if page["title"].startswith(("List of", "Lists of")):
                continue
            yield page


def fetch_article(session: requests.Session, pageid: int) -> tuple[str, dict]:
    """Return an article's plain text and the revision it came from."""
    data = api_query(
        session, pageids=pageid, prop="extracts|revisions",
        explaintext=1, exsectionformat="wiki", rvprop="ids|timestamp",
    )
    page = data["query"]["pages"][0]
    return page.get("extract", ""), page["revisions"][0]


def render(paragraphs: list[list[str]]) -> str:
    return "\n\n".join(" ".join(sentences) for sentences in paragraphs if sentences)


def cut_to_length(paragraphs: list[str], count_tokens, target: int) -> str:
    """Keep whole sentences from the start until the next one would pass `target` tokens."""
    kept: list[list[str]] = []
    for paragraph in paragraphs:
        current: list[str] = []
        for sentence in split_sentences(paragraph):
            if count_tokens(render(kept + [current + [sentence]])) > target:
                return render(kept + [current])
            current.append(sentence)
        kept.append(current)
    return render(kept)


def count_sentences(text: str) -> int:
    return sum(len(split_sentences(p)) for p in text.split("\n\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the carrier documents from Wikipedia.")
    parser.add_argument("--overwrite", action="store_true", help="delete existing documents and start again")
    parser.add_argument("--n", type=int, help="number of documents (default: from config)")
    parser.add_argument("--output-dir", type=Path, help="where to write (default: from config)")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())["documents"]
    n_documents = args.n or config["n_documents"]
    out_dir = args.output_dir or ROOT / config["output_dir"]
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        manifest_path.unlink(missing_ok=True)

    # Resume: keep documents already in the manifest. Document files without a
    # manifest row have no record of their source, so they are removed.
    rows = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    kept_files = {f"{row['doc_id']}.txt" for row in rows}
    orphans = [p for p in out_dir.glob("doc_*.txt") if p.name not in kept_files]
    for orphan in orphans:
        orphan.unlink()
    if orphans:
        print(f"Removed {len(orphans)} document file(s) with no manifest entry.")
    if len(rows) >= n_documents:
        sys.exit(f"Already have {len(rows)} documents in {out_dir}. Use --overwrite to start again.")
    if rows:
        print(f"Resuming: {len(rows)} documents already saved.")
    else:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    session = make_session()

    seen = {int(row["pageid"]) for row in rows}
    print(f"Collecting {n_documents} documents of {config['min_tokens']}–{config['target_tokens']} tokens...")
    for page in random_candidates(session, config["min_article_bytes"]):
        if page["pageid"] in seen:
            continue
        seen.add(page["pageid"])
        extract, revision = fetch_article(session, page["pageid"])
        text = cut_to_length(clean_extract(extract), count_tokens, config["target_tokens"])
        n_tokens = count_tokens(text)
        if n_tokens < config["min_tokens"]:
            print(f"   skipped  {page['title']} (only {n_tokens} tokens of prose)")
            continue

        doc_id = f"doc_{len(rows) + 1:02d}"
        (out_dir / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")
        row = {
            "doc_id": doc_id,
            "title": page["title"],
            "pageid": page["pageid"],
            "revid": revision["revid"],
            "revision_timestamp": revision["timestamp"],
            "url": f"https://en.wikipedia.org/w/index.php?oldid={revision['revid']}",
            "licence": LICENCE,
            "n_tokens": n_tokens,
            "n_sentences": count_sentences(text),
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with manifest_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow(row)
        rows.append(row)
        print(f"[{len(rows):2d}/{n_documents}] {page['title']} ({n_tokens} tokens, {row['n_sentences']} sentences)")
        if len(rows) == n_documents:
            break

    print(f"Done: {len(rows)} documents and manifest.csv in {out_dir}")


if __name__ == "__main__":
    main()
