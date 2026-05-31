# FH Invoice App

A Streamlit app for coding MRI AP invoice PDFs with Claude. Takes a multi-invoice
PDF batch, identifies invoice boundaries, predicts vendor / property / GL codes
grounded in your historical reference data, and lets you confirm and log them.

## How it works

1. **PDF extraction** — pdfplumber reads text per page. Pages whose text
   extraction failed (image-only scans or font-subset PDFs decoded as
   `(cid:N)` glyphs) are rendered to PNG for vision.
2. **Boundary detection** — one Sonnet 4.5 call reads the entire
   page-delimited document and returns structured invoice boundaries.
3. **Bundling** — pages are grouped per boundary. A merge/split UI lets
   you correct any errors before coding.
4. **Coding** — each bundle is sent to Sonnet 4.5 with the historical
   reference table as a cached system prompt. The model returns a
   structured prediction: vendor_code, property_code, gl_code, amount,
   description, and confidence.
5. **Review and save** — each prediction renders as an editable card.
   Confirmed predictions append to `coded_invoices.csv`.

Both Sonnet calls are content-addressed cached. Re-runs on the same PDF
are instant and free.

## Setup

```sh
pip install -r requirements.txt
```

### Configure your Anthropic API key (any one of these)

- **`.env` file** in the project root:

  ```
  ANTHROPIC_API_KEY=sk-ant-api03-...
  ```

- **Environment variable** before launching Streamlit:

  ```sh
  ANTHROPIC_API_KEY=sk-ant-api03-... streamlit run app.py
  ```

- **Streamlit secrets** — copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml` and paste your key. This is the recommended
  path for Streamlit Community Cloud deployment.

### Optional: configure the data directory

By default the app reads `invoice_reference.csv` and writes caches / logs
to the current working directory. Override with `DATA_DIR`:

```sh
DATA_DIR=/path/to/your/data streamlit run app.py
```

Files read/written under `DATA_DIR`:

| File | Purpose |
|---|---|
| `invoice_reference.csv` | Historical vendor / property / GL combinations (shipped with the repo) |
| `coded_invoices.csv` | Append-only log of confirmed predictions (created on first save) |
| `boundary_cache.json` | Cached invoice boundaries keyed by document text |
| `prediction_cache.json` | Cached coding predictions keyed by bundle content |
| `boundary_debug.txt` | Per-page text dump from the most recent upload (for debugging) |

### Run

```sh
streamlit run app.py
```

Upload a PDF, optionally adjust bundle boundaries with the merge / split
buttons, click **Analyze**, review each prediction, and save.

## Project structure

- `app.py` — Streamlit UI
- `pipeline.py` — pure logic: PDF extraction, boundary detection, coding,
  caching, split/merge helpers
- `test_batch.py` — CLI harness that runs the full pipeline against a
  PDF and prints the boundary list and predictions
- `invoice_reference.csv` — historical vendor / property / GL combinations
  with frequency counts (no amounts, no PII)

## Test against a local PDF

```sh
python test_batch.py /path/to/your/batch.pdf
```

## Models

- `claude-sonnet-4-5` for both boundary detection and coding
- Hybrid text + vision: image pages are rendered via `pdfplumber.Page.to_image()`
  and sent as base64 PNG content blocks alongside extracted text

## Cache invalidation

Deleting either cache file forces a fresh API call for that step.
Editing prompts in `pipeline.py` also invalidates the relevant cache
automatically (the cache key includes a hash of the prompt).
