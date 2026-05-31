"""Pure (Streamlit-free) pipeline functions.

Pipeline:
  1. pdfplumber extracts text per page (text only — never images / base64).
  2. claude-sonnet-4-5 reads the full page-delimited document in ONE call
     and returns invoice boundaries as structured JSON.
  3. group_bundles builds bundles directly from those boundaries.
  4. claude-sonnet-4-5 codes each bundle against the historical reference.

Two on-disk caches keep re-runs free:
  - boundary_cache.json:   document-text-keyed boundary results
  - prediction_cache.json: bundle-text-keyed coding predictions

Shared by app.py and test_batch.py.
"""

import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Literal

import anthropic
import pdfplumber
from pydantic import BaseModel, Field

CODING_MODEL = "claude-sonnet-4-5"
MODEL = CODING_MODEL  # back-compat for callers that imported MODEL
BOUNDARY_MODEL = CODING_MODEL
MAX_OUTPUT_TOKENS = 4096
BOUNDARY_MAX_OUTPUT_TOKENS = 8192
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.cwd()))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
BOUNDARY_CACHE_PATH = DATA_DIR / "boundary_cache.json"
PREDICTION_CACHE_PATH = DATA_DIR / "prediction_cache.json"


class InvoicePrediction(BaseModel):
    vendor_code: str = Field(description="Vendor code, e.g. FDH001")
    vendor_name: str = Field(description="Full vendor name from the invoice")
    invoice_number: str = Field(description="Invoice number as printed on the invoice")
    property_code: str = Field(description="Property code, e.g. 277")
    gl_code: str = Field(description="7-digit GL account code")
    gl_description: str = Field(description="GL account description")
    amount: float = Field(description="Total invoice amount in dollars")
    description: str = Field(description="One sentence describing what the invoice is for")
    confidence: Literal["high", "uncertain"]
    confidence_reason: str = Field(description="Brief reason for the confidence rating")


class InvoiceBoundary(BaseModel):
    start_page: int = Field(description="1-indexed starting page of this invoice")
    end_page: int = Field(description="1-indexed ending page of this invoice, inclusive (== start_page for single-page invoices)")
    vendor_hint: str = Field(description="Vendor name visible on the invoice, or empty string")
    invoice_number: str = Field(description="Invoice number as printed on the invoice, or empty string")


class InvoiceBoundaries(BaseModel):
    invoices: list[InvoiceBoundary] = Field(
        description="Every invoice in document order, with no gaps and no overlaps. The first must have start_page == 1; the last must have end_page == total page count."
    )


# ----- CID-encoded page detection (independent of boundary detection) -----

CID_RE = re.compile(r"\(cid:\d+\)")
CID_UNREADABLE_THRESHOLD = 0.90


def is_unreadable_page(text: str) -> bool:
    """True when more than 90% of extracted characters are (cid:N) codes."""
    if not text.strip():
        return False
    cid_total = sum(len(m) for m in CID_RE.findall(text))
    return cid_total / max(len(text), 1) > CID_UNREADABLE_THRESHOLD


def is_unreadable_bundle(bundle: list[tuple[int, str]]) -> bool:
    return len(bundle) == 1 and is_unreadable_page(bundle[0][1])


# ----- PDF extraction -----

def extract_pages(file_bytes: bytes) -> list[str]:
    """Per-page text. Text only — pdfplumber never returns image data here."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pages.append((p.extract_text() or "").replace("\x00", ""))
    return pages


# ----- Sonnet-based whole-document boundary detection -----

BOUNDARY_SYSTEM = """You are identifying invoice boundaries in a multi-invoice PDF batch from First Holding Management's accounts payable workflow.

The PDF is a scanned batch that concatenates many separate invoices into a single file. Pages are delimited in the input by "--- PAGE N ---" markers. Most pages appear as extracted text. Pages whose text extraction failed (image-only scans or pages with embedded font subsets) are provided as PAGE IMAGES instead — read those visually to identify the vendor letterhead, the bold "Invoice" header, and the invoice number. The batch is a mix of:

- INTERNAL LAWN MAINTENANCE SCHEDULES: single-page summaries (e.g., "Lawn Maintenance/Landscaping Monthly Payments 2026"). Each is its own primary invoice — one per property. A schedule MAY be followed by an optional VENDOR BACKUP PAGE (the contractor's actual invoice, e.g., "Invoice UW 707390" from United Lawnscape) attached behind it. That backup belongs to the schedule it sits behind, NOT a separate invoice.

- CONTRACTOR INVOICES for one-off work (Butcher & Butcher Construction, CertaSite, NBS Cleaning, etc.). These often span multiple pages and may be followed by supporting documents — proposals, estimates, work orders, photos, handwritten notes — all of which BELONG TO the preceding invoice.

- MULTI-PAGE UTILITY STATEMENTS from AT&T, Comcast, FedEx, etc. — typically 2 or 3 pages marked "Page: 1 of N", "Page: 2 of N", etc. Each statement is one invoice spanning all of its pages.

- SINGLE-PAGE INVOICES: standalone billings (single-page FedEx receipts, professional services bills, employee reimbursements). Each is its own invoice.

RULES:
1. INVOICE NUMBER is the strongest signal. Two pages with different invoice numbers are different invoices, regardless of shared template or vendor.
2. SAME TEMPLATE with DIFFERENT property name or DIFFERENT total = SEPARATE invoices. Two lawn-maintenance schedules side by side for different properties are two invoices, not one.
3. SUPPORTING DOCUMENTS — proposals, work orders, estimates, quotes, photos, handwritten notes, payment-schedule vendor backup — belong to the PRECEDING invoice. Never start a new invoice on a support page.
4. BLANK PAGES belong to the preceding invoice unless they clearly sit between two unrelated vendors.
5. COMPLETE COVERAGE: every page must belong to exactly one invoice. The first invoice MUST start at page 1; the last invoice MUST end at the total page count. No gaps. No overlaps.

LAWN SCHEDULE RULE (hard rule, no exceptions):
- Every lawn-maintenance schedule page ("Lawn Maintenance/Landscaping Monthly Payments 2026" etc.) is ALWAYS its own invoice. Single page. Never combined with adjacent schedule pages.
- Two consecutive lawn schedule pages with different contractor names (e.g., page 1 lists "DJ's Lawn Service", page 2 lists "Phoenix Landscape", page 3 lists "United Lawnscape") are ALWAYS three separate invoices, never one multi-page document.
- The ONLY case where a lawn schedule page bundles with the page immediately after it: when the next page is a United Lawnscape "Invoice UW XXXXXX" PDF (vendor letterhead saying "United Lawnscape" with an invoice number starting "UW "). In that single case, the UW invoice page is BACKUP attached to the lawn schedule directly before it.
- A lawn schedule page followed by another lawn schedule page → two separate invoices.
- A lawn schedule page followed by anything other than a "UW " United Lawnscape invoice → the lawn schedule is one invoice, the next page starts whatever it starts.

For each invoice, return:
- start_page: 1-indexed starting page
- end_page: 1-indexed ending page, inclusive (single-page invoices have end_page == start_page)
- vendor_hint: vendor name as visible on the invoice (e.g., "United Lawnscape", "AT&T", "Comcast Business"), empty string if unclear
- invoice_number: invoice number as printed (e.g., "UW 707390", "9-314-03120"), empty string if absent

Return all invoices in document order."""


def _boundary_prompt_version() -> str:
    """Short hash of the boundary system prompt + model. Prompt edits invalidate cache."""
    return hashlib.sha256(
        (BOUNDARY_SYSTEM + "\x1f" + BOUNDARY_MODEL).encode("utf-8")
    ).hexdigest()[:12]


def _boundary_cache_key(pages: list[str]) -> str:
    """Key on the full document text + prompt version."""
    sep = "\x1f"
    payload = sep.join([_boundary_prompt_version(), *pages])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_boundary_cache(path: Path = BOUNDARY_CACHE_PATH) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_boundary_cache(cache: dict[str, list[dict]], path: Path = BOUNDARY_CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


VISION_CID_THRESHOLD = 0.40
VISION_RENDER_DPI = 150


def _needs_vision(text: str) -> bool:
    """True when pdfplumber returned no text or text is >40% (cid:N) glyphs.

    These pages need to be sent to Sonnet as page images so it can read them
    visually instead of from the failed text extraction.
    """
    if not text.strip():
        return True
    cid_chars = sum(len(m) for m in CID_RE.findall(text))
    return cid_chars / max(len(text), 1) > VISION_CID_THRESHOLD


def _render_pages_as_b64_png(file_bytes: bytes, page_nums: list[int]) -> dict[int, str]:
    """Render the requested 1-indexed pages as PNG and return {page_num: base64 str}."""
    if not page_nums:
        return {}
    result: dict[int, str] = {}
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for n in page_nums:
            if n < 1 or n > len(pdf.pages):
                continue
            page = pdf.pages[n - 1]
            img = page.to_image(resolution=VISION_RENDER_DPI)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result[n] = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return result


def _build_boundary_content(pages: list[str], file_bytes: bytes | None) -> list[dict]:
    """Build the user-message content list: text blocks for normal pages,
    page image blocks for pages where text extraction failed."""
    if file_bytes is None:
        body = "\n\n".join(f"--- PAGE {i + 1} ---\n{text}" for i, text in enumerate(pages))
        return [{"type": "text", "text": body}]

    vision_pages = sorted(i + 1 for i, text in enumerate(pages) if _needs_vision(text))
    if not vision_pages:
        body = "\n\n".join(f"--- PAGE {i + 1} ---\n{text}" for i, text in enumerate(pages))
        return [{"type": "text", "text": body}]

    images = _render_pages_as_b64_png(file_bytes, vision_pages)
    vision_set = set(vision_pages)

    content: list[dict] = []
    for i, text in enumerate(pages):
        page_num = i + 1
        if page_num in vision_set and page_num in images:
            content.append({"type": "text", "text": f"\n--- PAGE {page_num} (image) ---\n"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": images[page_num],
                },
            })
        else:
            content.append({"type": "text", "text": f"\n--- PAGE {page_num} ---\n{text}\n"})
    return content


def find_invoice_boundaries(
    client: anthropic.Anthropic,
    pages: list[str],
    file_bytes: bytes | None = None,
) -> InvoiceBoundaries:
    """Single Sonnet call that reads the page-delimited document and returns invoice boundaries.

    When file_bytes is provided, pages whose text extraction failed (empty or
    >40% CID glyphs) are rendered to PNG and included as image content blocks
    so Sonnet can read them visually.

    Cached by document text + prompt version, so re-runs on the same PDF are free.
    """
    cache = _load_boundary_cache()
    key = _boundary_cache_key(pages)
    cached = cache.get(key)
    if cached is not None:
        return InvoiceBoundaries.model_validate({"invoices": cached})

    content = _build_boundary_content(pages, file_bytes)
    response = client.messages.parse(
        model=BOUNDARY_MODEL,
        max_tokens=BOUNDARY_MAX_OUTPUT_TOKENS,
        temperature=0,
        system=BOUNDARY_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=InvoiceBoundaries,
    )
    boundaries = response.parsed_output
    cache[key] = [b.model_dump() for b in boundaries.invoices]
    _save_boundary_cache(cache)
    return boundaries


# ----- Bundling -----

def group_bundles(
    page_texts: list[str],
    boundaries: InvoiceBoundaries | list[InvoiceBoundary],
) -> list[list[tuple[int, str]]]:
    """Build bundles from Sonnet's invoice boundaries.

    Each bundle is the list of (page_num, page_text) tuples for one invoice.
    Pages are clamped to [1, len(page_texts)]; overlapping ranges are repaired
    by truncating the earlier (longer) range so each page belongs to exactly
    one bundle.
    """
    if isinstance(boundaries, InvoiceBoundaries):
        boundaries_list = boundaries.invoices
    else:
        boundaries_list = boundaries

    sorted_b = sorted(boundaries_list, key=lambda b: (b.start_page, b.end_page))
    ranges: list[list[int]] = []  # mutable [start, end] pairs
    for b in sorted_b:
        start = max(1, b.start_page)
        end = min(len(page_texts), b.end_page)
        if start <= end:
            ranges.append([start, end])

    # If two adjacent ranges overlap, truncate the earlier one to end one
    # page before the later one starts. The later boundary "wins" because
    # Sonnet flagged that page as the start of a different invoice.
    for i in range(len(ranges) - 1):
        if ranges[i][1] >= ranges[i + 1][0]:
            ranges[i][1] = ranges[i + 1][0] - 1

    bundles: list[list[tuple[int, str]]] = []
    for start, end in ranges:
        if start > end:
            continue
        bundle = [(p, page_texts[p - 1]) for p in range(start, end + 1)]
        bundles.append(bundle)
    return bundles


def bundle_label(bundle: list[tuple[int, str]]) -> str:
    if len(bundle) == 1:
        return f"page {bundle[0][0]}"
    return f"pages {bundle[0][0]}–{bundle[-1][0]}"


def bundle_preview_snippet(bundle: list[tuple[int, str]], n: int = 80) -> str:
    for _, text in bundle:
        normalized = " ".join(text.split())
        if normalized:
            return normalized[:n]
    return ""


def split_bundle(
    bundles: list[list[tuple[int, str]]],
    bundle_idx: int,
    split_after_page: int,
) -> list[list[tuple[int, str]]]:
    if bundle_idx < 0 or bundle_idx >= len(bundles):
        return bundles
    bundle = bundles[bundle_idx]
    split_at: int | None = None
    for i, (page_num, _) in enumerate(bundle):
        if page_num == split_after_page:
            split_at = i + 1
            break
    if split_at is None or split_at >= len(bundle):
        return bundles
    new_bundles = list(bundles)
    new_bundles[bundle_idx : bundle_idx + 1] = [bundle[:split_at], bundle[split_at:]]
    return new_bundles


def merge_bundle_with_next(
    bundles: list[list[tuple[int, str]]],
    bundle_idx: int,
) -> list[list[tuple[int, str]]]:
    if bundle_idx < 0 or bundle_idx >= len(bundles) - 1:
        return bundles
    new_bundles = list(bundles)
    merged = new_bundles[bundle_idx] + new_bundles[bundle_idx + 1]
    new_bundles[bundle_idx : bundle_idx + 2] = [merged]
    return new_bundles


def write_boundary_debug(
    source_name: str,
    pages: list[str],
    boundaries: InvoiceBoundaries | list[InvoiceBoundary],
    out_path,
) -> None:
    """Write a per-page text snippet + boundary listing to a text file."""
    if isinstance(boundaries, InvoiceBoundaries):
        boundaries_list = boundaries.invoices
    else:
        boundaries_list = boundaries
    bundles = group_bundles(pages, boundaries)
    lines: list[str] = []
    lines.append(f"File:    {source_name}")
    lines.append(f"Pages:   {len(pages)}")
    lines.append(f"Bundles: {len(bundles)}")
    lines.append("")
    lines.append("Per-page text snippet:")
    lines.append("-" * 90)
    for i, text in enumerate(pages, 1):
        snippet = " ".join(text.split())[:70]
        lines.append(f"Page {i:>3}: {snippet}")
    lines.append("")
    lines.append("Invoice boundaries (sonnet):")
    for i, b in enumerate(boundaries_list, 1):
        if b.start_page == b.end_page:
            page_range = f"page {b.start_page}"
        else:
            page_range = f"pages {b.start_page}-{b.end_page}"
        vendor = b.vendor_hint or "(no vendor)"
        inv = b.invoice_number or "(no inv#)"
        lines.append(f"  Invoice {i:>2}: {page_range:<14s}  vendor={vendor:<32s}  invoice_number={inv}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----- Reference data + Sonnet coding call (with prediction cache) -----

def load_reference_table(path: Path) -> str:
    return path.read_text(encoding="utf-8")


SYSTEM_INSTRUCTIONS = """You code invoices for First Holding Management Company's payables team.

You will receive the extracted text of a SINGLE invoice. Use the historical
reference table below to predict vendor_code, property_code, and gl_code.

The reference table shows, for every (vendor, property, GL) combination ever
used, the most common GL description and how often that combination has
appeared. Use it as your primary source of truth.

PRIORITY RULES — apply these BEFORE the per-field heuristics, in the
order listed. Rule 1B is the strongest and overrides all others.

RULE 1B — EXACT ACCOUNT/INVOICE NUMBER MATCH (strongest signal of all):
Before applying any other rule, search the historical reference table
for the invoice number, account number, or any unique identifier
appearing on the current invoice page. Look in the gl_description
column — historical rows often embed the account number there, e.g.
"052024-061924A#318422949" is a description that contains AT&T
account 318422949. If you find one or more historical rows whose
description contains the current invoice's number/account, use THAT
row's vendor_code, property_code, and gl_code without deviation.
This is the strongest possible signal and overrides Rule 1, Rule 2,
and all per-field heuristics. When multiple historical rows match the
same account number, they will agree on property and GL — use that
agreed pair. An exact account-number match in history beats vendor
frequency, named-entity inference, and your own classification.

RULE 1 — REFERENCE DATA PRIORITY (high-frequency history wins):
When the vendor_code you identify appears in the reference with a count of
5 or more for some (vendor, property, GL) combination, you MUST use that
historical gl_code and property_code as your default prediction. High-
frequency vendor history is more reliable than your own inference. Deviate
from this default ONLY when the current invoice EXPLICITLY contradicts it
— for example, the invoice text clearly names a different property than
the historical pattern shows, or the work described is unambiguously a
different expense category than the historical GL. When the invoice is
silent or ambiguous, trust the history with count ≥ 5.

RULE 2 — NAMED ENTITY PRIORITY (explicit names on the invoice win):
When a property name, entity name, or company name appears explicitly on
the invoice page — in the billing address, page header, "Bill To" field,
"Service Location" field, or anywhere else on the page — that named entity
determines the property_code. Named entities on the invoice ALWAYS override
historical inference. Example: "Rochester KM Partners" anywhere on the
page means property 295, full stop, regardless of what the historical
pattern for this vendor might otherwise suggest.

Per-field rules (apply after the priority rules above):
- vendor_code: match by vendor name against the reference; the reference's
  vendor_code is authoritative
- property_code: apply Rule 2 first (named entity on the invoice wins).
  Otherwise infer from billing address, account number, or property name;
  favor combinations that exist in the reference for this vendor
- gl_code: apply Rule 1 first (≥5-count historical GL wins). Otherwise
  classify the expense type (utilities, repairs, permits, etc.) against
  historical patterns for this vendor + property
- amount: the FINAL TOTAL of the invoice — the amount the payee owes after
  all line items, taxes, fees, surcharges, discounts, and credits have been
  applied. Look for labels like "Total", "Total Due", "Amount Due", "Grand
  Total", "Balance Due", "Invoice Total", "New Charges", or "Please Pay".
  This is typically printed once near the bottom or top of the invoice and
  should equal the sum of all line items plus tax. For invoices with multiple
  line items, the amount is the SUM of all lines (with tax/fees) — NEVER a
  single line item, NEVER the largest line item, NEVER the subtotal (which
  excludes tax). If the printed total is unclear or missing, sum the
  individual line items yourself (including all taxes, fees, and surcharges
  shown) and report that sum.
- invoice_number: as printed on the invoice
- description: one sentence on what was billed

Confidence:
- "high": vendor + property + GL combination exists in the reference
  (especially with count > 1), or the invoice text is unambiguous
- "uncertain": new vendor, property cannot be determined, GL doesn't match
  historical patterns for this vendor, or invoice text is ambiguous

Always return valid values for every field. If a field truly cannot be
determined, return an empty string and explain in confidence_reason."""


ZERO_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "output_tokens": 0,
}


def _coding_prompt_version() -> str:
    """Hash of the Sonnet system prompt + model id. Prompt edits invalidate cache."""
    return hashlib.sha256(
        (SYSTEM_INSTRUCTIONS + "\x1f" + CODING_MODEL).encode("utf-8")
    ).hexdigest()[:12]


def _prediction_cache_key(
    bundle: list[tuple[int, str]],
    reference_table: str,
    page_images: dict[int, str] | None = None,
) -> str:
    """Key on the bundle's per-page content + reference + prompt version.

    For pages with extractable text, the key uses the text. For pages that
    were sent to Sonnet as images, the key uses a hash of the image bytes
    so two image-only pages with the same empty extracted text don't collide.

    Page numbers and filename are excluded so the same invoice content
    returns the same prediction regardless of which physical pages it
    occupied or which PDF it came from.
    """
    page_images = page_images or {}
    sep = "\x1f"
    parts: list[str] = []
    for pn, text in bundle:
        if pn in page_images:
            img_hash = hashlib.sha256(page_images[pn].encode("ascii")).hexdigest()[:16]
            parts.append(f"IMG:{img_hash}")
        else:
            parts.append(text)
    body = sep.join(parts)
    ref_version = hashlib.sha256(reference_table.encode("utf-8")).hexdigest()[:12]
    payload = sep.join([_coding_prompt_version(), ref_version, body])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_bundle_vision_pages(file_bytes: bytes, bundle: list[tuple[int, str]]) -> dict[int, str]:
    """Render base64 PNG for bundle pages that need vision (empty or CID-heavy)."""
    vision_pages = [pn for pn, text in bundle if _needs_vision(text)]
    if not vision_pages:
        return {}
    return _render_pages_as_b64_png(file_bytes, vision_pages)


def _build_predict_one_content(
    bundle: list[tuple[int, str]],
    filename: str,
    page_images: dict[int, str],
) -> list[dict]:
    """Build user message content: text blocks for normal pages, image blocks
    for pages whose text extraction failed."""
    if page_images:
        intro_text = (
            f"PDF filename: {filename}\n"
            f"This single invoice spans {bundle_label(bundle)} of the source PDF.\n\n"
            "Per-page content (text where extracted, page images where text extraction failed):\n"
        )
    else:
        intro_text = (
            f"PDF filename: {filename}\n"
            f"This single invoice spans {bundle_label(bundle)} of the source PDF.\n\n"
            "Extracted text (text only — no images):\n"
        )
    content: list[dict] = [{"type": "text", "text": intro_text}]
    for pn, text in bundle:
        if pn in page_images:
            content.append({"type": "text", "text": f"\n--- Page {pn} (image) ---\n"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": page_images[pn],
                },
            })
        else:
            content.append({"type": "text", "text": f"\n--- Page {pn} ---\n{text}\n"})
    return content


def _load_prediction_cache(path: Path = PREDICTION_CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_prediction_cache(cache: dict[str, dict], path: Path = PREDICTION_CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# Hand-written strict schema for Anthropic's structured output. Matches the
# InvoicePrediction Pydantic model but adds "additionalProperties": false
# and lists every field in "required", as the API requires.
INVOICE_PREDICTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "vendor_code": {"type": "string", "description": "Vendor code, e.g. FDH001"},
        "vendor_name": {"type": "string", "description": "Full vendor name from the invoice"},
        "invoice_number": {"type": "string", "description": "Invoice number as printed on the invoice"},
        "property_code": {"type": "string", "description": "Property code, e.g. 277"},
        "gl_code": {"type": "string", "description": "7-digit GL account code"},
        "gl_description": {"type": "string", "description": "GL account description"},
        "amount": {"type": "number", "description": "Total invoice amount in dollars"},
        "description": {"type": "string", "description": "One sentence describing what the invoice is for"},
        "confidence": {"type": "string", "enum": ["high", "uncertain"]},
        "confidence_reason": {"type": "string", "description": "Brief reason for the confidence rating"},
    },
    "required": [
        "vendor_code", "vendor_name", "invoice_number", "property_code",
        "gl_code", "gl_description", "amount", "description",
        "confidence", "confidence_reason",
    ],
    "additionalProperties": False,
}

_PREDICTION_FIELD_NAMES = [
    "vendor_code", "vendor_name", "invoice_number", "property_code",
    "gl_code", "gl_description", "amount", "description",
    "confidence", "confidence_reason",
]


def _extract_json_fields(raw: str) -> dict:
    """Best-effort extraction of field values from a possibly-truncated JSON object.

    Walks the field list with a regex that matches a complete string literal
    or number. Partial strings (the truncation point) are skipped silently.
    """
    fields: dict = {}
    for name in _PREDICTION_FIELD_NAMES:
        pattern = rf'"{re.escape(name)}"\s*:\s*("(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?)'
        m = re.search(pattern, raw)
        if not m:
            continue
        val_str = m.group(1)
        if val_str.startswith('"'):
            try:
                fields[name] = json.loads(val_str)
            except json.JSONDecodeError:
                pass
        else:
            try:
                fields[name] = float(val_str)
            except ValueError:
                pass
    return fields


def _repair_invoice_prediction(raw_text: str) -> InvoicePrediction:
    """Build a partial InvoicePrediction from a truncated/malformed JSON response.

    Always returns a valid InvoicePrediction with confidence='uncertain' so
    downstream code can proceed; missing fields default to empty/zero and
    confidence_reason explains the recovery.
    """
    fields = _extract_json_fields(raw_text)
    amount_val = fields.get("amount")
    if isinstance(amount_val, (int, float)):
        amount = float(amount_val)
    else:
        amount = 0.0
    confidence_raw = str(fields.get("confidence", "")).lower()
    confidence: Literal["high", "uncertain"] = "high" if confidence_raw == "high" else "uncertain"
    recovered = sorted(fields.keys())
    return InvoicePrediction(
        vendor_code=str(fields.get("vendor_code", "")),
        vendor_name=str(fields.get("vendor_name", "")),
        invoice_number=str(fields.get("invoice_number", "")),
        property_code=str(fields.get("property_code", "")),
        gl_code=str(fields.get("gl_code", "")),
        gl_description=str(fields.get("gl_description", "")),
        amount=amount,
        description=str(fields.get("description", "")),
        confidence="uncertain",  # always uncertain on repair, even if partial said "high"
        confidence_reason=(
            f"Response was truncated or malformed. Recovered {len(recovered)} "
            f"field(s) from the partial JSON: {', '.join(recovered) or 'none'}. "
            "Please review and correct."
        ),
    )


def predict_one(
    client: anthropic.Anthropic,
    bundle: list[tuple[int, str]],
    filename: str,
    reference_table: str,
    file_bytes: bytes | None = None,
) -> tuple[InvoicePrediction, dict]:
    """Code one invoice bundle with Sonnet, with content-addressed caching.

    When file_bytes is provided and any page in the bundle has failed text
    extraction (empty or >40% CID glyphs), those pages are rendered to PNG
    and sent as image content blocks so Sonnet can read them visually.

    Cache hits return the stored prediction with zero usage so token totals
    only reflect API calls actually made.
    """
    page_images = _render_bundle_vision_pages(file_bytes, bundle) if file_bytes else {}

    cache = _load_prediction_cache()
    key = _prediction_cache_key(bundle, reference_table, page_images)
    cached_val = cache.get(key)
    if cached_val is not None:
        return InvoicePrediction.model_validate(cached_val), dict(ZERO_USAGE)

    system_blocks = [
        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"HISTORICAL REFERENCE TABLE (CSV):\n{reference_table}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    content = _build_predict_one_content(bundle, filename, page_images)

    response = client.with_options(max_retries=8).messages.create(
        model=CODING_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,
        system=system_blocks,
        messages=[{"role": "user", "content": content}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": INVOICE_PREDICTION_SCHEMA,
            }
        },
    )

    usage = {
        "input_tokens": response.usage.input_tokens,
        "cache_read_input_tokens": response.usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": response.usage.cache_creation_input_tokens or 0,
        "output_tokens": response.usage.output_tokens,
    }

    raw_text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        prediction = InvoicePrediction.model_validate_json(raw_text)
    except Exception:
        prediction = _repair_invoice_prediction(raw_text)

    cache[key] = prediction.model_dump()
    _save_prediction_cache(cache)
    return prediction, usage
