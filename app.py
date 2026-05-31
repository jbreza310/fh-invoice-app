"""Invoice coding app for First Holding payables.

UI shell + log persistence. All PDF parsing / heuristic bundling / Claude
calls live in pipeline.py so the same logic is exercised by test_batch.py.
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline import (
    BOUNDARY_MODEL,
    CODING_MODEL,
    bundle_label,
    bundle_preview_snippet,
    extract_pages,
    find_invoice_boundaries,
    group_bundles,
    is_unreadable_bundle,
    load_reference_table,
    merge_bundle_with_next,
    predict_one,
    split_bundle,
    write_boundary_debug,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.cwd()))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(DATA_DIR / ".env")

REFERENCE_CSV = DATA_DIR / "invoice_reference.csv"
LOG_CSV = DATA_DIR / "coded_invoices.csv"
BOUNDARY_DEBUG_TXT = DATA_DIR / "boundary_debug.txt"

LOG_FIELDS = [
    "date_coded",
    "vendor_code",
    "vendor_name",
    "invoice_number",
    "property_code",
    "gl_code",
    "gl_description",
    "amount",
    "source_file",
    "confirmed_by",
]


@st.cache_resource
def get_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError, AttributeError):
            pass
    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY is not set. Provide it via .env, the "
            "ANTHROPIC_API_KEY env var, or .streamlit/secrets.toml."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_data
def cached_reference_table() -> str:
    if not REFERENCE_CSV.exists():
        st.error(f"Reference file not found: {REFERENCE_CSV}")
        st.stop()
    return load_reference_table(REFERENCE_CSV)


def predict_all(client, bundles, filename: str, file_bytes: bytes | None = None):
    reference_table = cached_reference_table()
    predictions = []
    errors = []
    totals = dict.fromkeys(
        ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"], 0
    )
    progress = st.progress(0.0, text=f"Coding 0/{len(bundles)} invoices...")
    for i, bundle in enumerate(bundles, 1):
        if is_unreadable_bundle(bundle):
            predictions.append(None)
            progress.progress(i / len(bundles), text=f"Coded {i}/{len(bundles)} invoices")
            continue
        try:
            pred, usage = predict_one(client, bundle, filename, reference_table, file_bytes=file_bytes)
            predictions.append(pred)
            for k in totals:
                totals[k] += usage[k]
        except Exception as e:
            errors.append((i, str(e)))
            predictions.append(None)
        progress.progress(i / len(bundles), text=f"Coded {i}/{len(bundles)} invoices")
    progress.empty()
    return predictions, totals, errors


def append_to_log(row: dict) -> None:
    is_new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_log_tail(n: int = 10) -> list[dict]:
    if not LOG_CSV.exists():
        return []
    with LOG_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:][::-1]


def reset_upload_state() -> None:
    st.session_state.bundles = None
    st.session_state.predictions = None
    st.session_state.confirmed_idx = set()
    st.session_state.pdf_pages = None
    st.session_state.pdf_name = None
    st.session_state.pdf_file_bytes = None
    st.session_state.last_usage = None


# ----- UI -----

st.set_page_config(page_title="FH Invoice Coding", layout="wide")
st.title("Invoice Coding")
st.caption(
    f"Boundary detection: {BOUNDARY_MODEL}  ·  Coding: {CODING_MODEL}  ·  "
    f"Reference: {REFERENCE_CSV.name}  ·  Log: {LOG_CSV.name}"
)

client = get_client()

for k, default in [
    ("bundles", None),
    ("predictions", None),
    ("confirmed_idx", set()),
    ("pdf_pages", None),
    ("pdf_name", None),
    ("pdf_file_bytes", None),
    ("last_usage", None),
]:
    if k not in st.session_state:
        st.session_state[k] = default

uploaded = st.file_uploader(
    "Upload an invoice PDF (one file, may contain multiple invoices)",
    type=["pdf"],
)

if uploaded is not None and uploaded.name != st.session_state.pdf_name:
    file_bytes = uploaded.getvalue()
    with st.spinner("Extracting page text..."):
        pages = extract_pages(file_bytes)

    with st.spinner(f"Identifying invoice boundaries with {BOUNDARY_MODEL} (hybrid text + vision)..."):
        boundaries = find_invoice_boundaries(client, pages, file_bytes=file_bytes)

    bundles = group_bundles(pages, boundaries)
    write_boundary_debug(uploaded.name, pages, boundaries, BOUNDARY_DEBUG_TXT)
    st.session_state.pdf_pages = pages
    st.session_state.bundles = bundles
    st.session_state.pdf_name = uploaded.name
    st.session_state.pdf_file_bytes = file_bytes
    st.session_state.predictions = None
    st.session_state.confirmed_idx = set()
    st.session_state.last_usage = None

def invalidate_predictions_after_edit() -> None:
    """When bundles are restructured, predictions and saves no longer line up
    with the new bundle indices, so clear them and force a fresh Analyze."""
    st.session_state.predictions = None
    st.session_state.confirmed_idx = set()
    st.session_state.last_usage = None


if st.session_state.bundles is not None:
    pages = st.session_state.pdf_pages
    bundles = st.session_state.bundles
    preds = st.session_state.predictions
    saved = st.session_state.confirmed_idx

    top1, top2 = st.columns([2, 1])
    with top1:
        st.markdown(
            f"**{st.session_state.pdf_name}** — "
            f"{len(pages)} page(s) → {len(bundles)} bundle(s)"
        )
        if preds is None:
            st.caption(
                "Adjust bundle boundaries below if the auto-detection is off, "
                "then click Analyze."
            )
        with st.expander("Show extracted text per page"):
            for i, t in enumerate(pages, 1):
                st.markdown(f"**Page {i}**")
                st.text(t or "(no text extracted)")

    with top2:
        label = "Re-analyze" if preds else "Analyze with Claude"
        if st.button(label, type="primary", use_container_width=True):
            new_preds, totals, errors = predict_all(
                client,
                bundles,
                st.session_state.pdf_name,
                file_bytes=st.session_state.pdf_file_bytes,
            )
            st.session_state.predictions = new_preds
            st.session_state.confirmed_idx = set()
            st.session_state.last_usage = totals
            for idx, msg in errors:
                st.error(f"Bundle {idx} failed: {msg}")
            st.rerun()

    if preds and st.session_state.last_usage:
        u = st.session_state.last_usage
        st.caption(
            f"Tokens (sum across {len(preds)} call(s)) — "
            f"input {u['input_tokens']}, "
            f"cache read {u['cache_read_input_tokens']}, "
            f"cache write {u['cache_creation_input_tokens']}, "
            f"output {u['output_tokens']}"
        )

    st.divider()

    for i, bundle in enumerate(bundles):
        pred = preds[i] if preds else None

        with st.container(border=True):
            page_chip = bundle_label(bundle)
            header_suffix = ""
            if pred is not None:
                header_suffix = f"  ·  {pred.vendor_name or '(no vendor)'}  ·  ${pred.amount:,.2f}"
            elif pred is None and is_unreadable_bundle(bundle):
                header_suffix = "  ·  (unreadable scan)"
            st.markdown(f"#### #{i + 1}  ·  {page_chip}{header_suffix}")

            snippet = bundle_preview_snippet(bundle)
            if snippet:
                st.caption(f"→ {snippet}")

            unreadable = pred is None and is_unreadable_bundle(bundle)

            if unreadable:
                st.warning(
                    "This page could not be read — PDF text decodes mostly as `(cid:N)` "
                    "glyphs (typical for scanned images with embedded font subsets). "
                    "If it actually belongs to an adjacent invoice, mark it as readable below "
                    "to merge it in; otherwise code it manually from the source PDF."
                )
                override_cols = st.columns([3, 3, 4])
                if i > 0:
                    if override_cols[0].button(
                        "Mark as readable & merge ↑",
                        key=f"mark_up_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.bundles = merge_bundle_with_next(bundles, i - 1)
                        invalidate_predictions_after_edit()
                        st.rerun()
                if i < len(bundles) - 1:
                    if override_cols[1].button(
                        "Mark as readable & merge ↓",
                        key=f"mark_down_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.bundles = merge_bundle_with_next(bundles, i)
                        invalidate_predictions_after_edit()
                        st.rerun()
                continue

            # Regular merge/split controls — always available on codeable cards.
            can_split = len(bundle) > 1
            can_merge = i < len(bundles) - 1
            if can_split or can_merge:
                ctrl_cols = st.columns([3, 2, 5])
                if can_split:
                    page_options = [page_num for page_num, _ in bundle[:-1]]
                    sub = ctrl_cols[0].columns([3, 2])
                    split_after = sub[0].selectbox(
                        "Split after page",
                        page_options,
                        key=f"split_sel_{i}",
                        label_visibility="collapsed",
                    )
                    if sub[1].button("Split here", key=f"split_btn_{i}", use_container_width=True):
                        st.session_state.bundles = split_bundle(bundles, i, split_after)
                        invalidate_predictions_after_edit()
                        st.rerun()
                if can_merge:
                    if ctrl_cols[1].button(
                        "Merge with next ↓",
                        key=f"merge_btn_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.bundles = merge_bundle_with_next(bundles, i)
                        invalidate_predictions_after_edit()
                        st.rerun()

            if pred is None:
                continue

            if i in saved:
                st.success("Saved.")
                continue

            if pred.confidence == "high":
                st.success(f"High confidence — {pred.confidence_reason}")
            else:
                st.warning(f"Uncertain — {pred.confidence_reason}")

            with st.form(f"form_{i}"):
                c1, c2 = st.columns(2)
                with c1:
                    vendor_code = st.text_input("Vendor code", pred.vendor_code, key=f"vc_{i}")
                    vendor_name = st.text_input("Vendor name", pred.vendor_name, key=f"vn_{i}")
                    invoice_number = st.text_input("Invoice number", pred.invoice_number, key=f"in_{i}")
                    amount = st.number_input(
                        "Amount", value=float(pred.amount), step=0.01, format="%.2f", key=f"am_{i}"
                    )
                with c2:
                    property_code = st.text_input("Property code", pred.property_code, key=f"pc_{i}")
                    gl_code = st.text_input("GL code", pred.gl_code, key=f"gc_{i}")
                    gl_description = st.text_input("GL description", pred.gl_description, key=f"gd_{i}")
                    confirmed_by = st.text_input(
                        "Confirmed by", os.environ.get("USERNAME", ""), key=f"cb_{i}"
                    )
                description = st.text_area("Description", pred.description, height=70, key=f"ds_{i}")

                if st.form_submit_button("Confirm & save", type="primary"):
                    append_to_log({
                        "date_coded": datetime.now().isoformat(timespec="seconds"),
                        "vendor_code": vendor_code,
                        "vendor_name": vendor_name,
                        "invoice_number": invoice_number,
                        "property_code": property_code,
                        "gl_code": gl_code,
                        "gl_description": gl_description,
                        "amount": amount,
                        "source_file": st.session_state.pdf_name,
                        "confirmed_by": confirmed_by,
                    })
                    st.session_state.confirmed_idx.add(i)
                    st.rerun()

    if preds:
        codeable = sum(1 for p in preds if p is not None)
        if len(saved) == codeable and codeable > 0:
            unreadable = len(preds) - codeable
            msg = f"All {codeable} codeable invoices saved."
            if unreadable:
                msg += f" {unreadable} unreadable page(s) flagged for manual review above."
            st.success(msg + " Upload another PDF to continue.")
            if st.button("Clear and upload another"):
                reset_upload_state()
                st.rerun()

with st.expander("Recent confirmed (last 10)"):
    rows = read_log_tail(10)
    if not rows:
        st.write("No confirmations yet.")
    else:
        st.dataframe(rows, use_container_width=True)
