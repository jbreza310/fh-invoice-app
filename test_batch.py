"""Run the full pipeline (Sonnet boundary detection + Sonnet coding) against
a batch PDF and print diagnostics. Reads ANTHROPIC_API_KEY from .env."""

import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from pipeline import (
    bundle_label,
    extract_pages,
    find_invoice_boundaries,
    group_bundles,
    load_reference_table,
    predict_one,
    write_boundary_debug,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.cwd()))).resolve()
REFERENCE_CSV = DATA_DIR / "invoice_reference.csv"
DEBUG_TXT = DATA_DIR / "boundary_debug.txt"

if len(sys.argv) < 2:
    print("Usage: python test_batch.py <pdf-path>")
    sys.exit(2)

pdf_path = Path(sys.argv[1])
if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    sys.exit(2)

load_dotenv(DATA_DIR / ".env")

print(f"File: {pdf_path}")
print(f"Size: {pdf_path.stat().st_size / 1024:.0f} KB")

file_bytes = pdf_path.read_bytes()
pages = extract_pages(file_bytes)
print(f"Pages: {len(pages)}\n")

client = anthropic.Anthropic()

# Report which pages will be sent as images.
from pipeline import _needs_vision  # noqa: E402
vision_pages = [i + 1 for i, t in enumerate(pages) if _needs_vision(t)]
if vision_pages:
    print(f"Pages needing vision (empty / >40% CID): {vision_pages}")

print("Identifying invoice boundaries with claude-sonnet-4-5 (hybrid text + vision)...")
boundaries = find_invoice_boundaries(client, pages, file_bytes=file_bytes)
bundles = group_bundles(pages, boundaries)

print(f"\nFound {len(boundaries.invoices)} invoice boundary record(s) → {len(bundles)} bundle(s):")
for i, b in enumerate(boundaries.invoices, 1):
    rng = f"page {b.start_page}" if b.start_page == b.end_page else f"pages {b.start_page}-{b.end_page}"
    vendor = b.vendor_hint or "(no vendor)"
    inv = b.invoice_number or "(no inv#)"
    print(f"  #{i:>2}: {rng:<14s}  vendor={vendor:<32s} invoice_number={inv}")

write_boundary_debug(pdf_path.name, pages, boundaries, DEBUG_TXT)
print(f"\nWrote {DEBUG_TXT}")

print()
print("Calling Claude (one Sonnet request per bundle)...")
print("-" * 100)

reference_table = load_reference_table(REFERENCE_CSV)
totals = dict.fromkeys(
    ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"], 0
)

for i, bundle in enumerate(bundles, 1):
    label = bundle_label(bundle)
    try:
        pred, usage = predict_one(client, bundle, pdf_path.name, reference_table, file_bytes=file_bytes)
    except Exception as e:
        print(f"  #{i:>2} {label:>15s} | FAILED: {e}")
        continue
    for k in totals:
        totals[k] += usage[k]
    print(
        f"  #{i:>2} {label:>15s} | {pred.vendor_code:>7s} {pred.vendor_name[:28]:<28s}"
        f" | prop={pred.property_code:>4s} gl={pred.gl_code:>7s} {pred.gl_description[:24]:<24s}"
        f" | ${pred.amount:>10,.2f} | {pred.confidence:>9s}"
    )

print()
print("Sonnet token totals (coding pass only):")
for k, v in totals.items():
    print(f"  {k:>32s}: {v:>8,}")

cost = (
    totals["input_tokens"] * 3e-6
    + totals["cache_read_input_tokens"] * 0.3e-6
    + totals["cache_creation_input_tokens"] * 3.75e-6
    + totals["output_tokens"] * 15e-6
)
print(f"  {'estimated Sonnet cost (USD)':>32s}: ${cost:.4f}")
