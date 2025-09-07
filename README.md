# NL2SQL — README

**Source file:** `untitled9.py`

## Short summary
This project runs a *local* NL→SQL pipeline using a Mistral Instruct model (quantized to 4-bit with bitsandbytes) and exposes a small Gradio UI. Upload a CSV, ask a natural-language question about the table, the local model generates an SQLite query, and the script runs it in-memory and returns results.

## Features
- Loads a 4-bit quantized LLM using `transformers` + `bitsandbytes`.
- Builds a prompt from the uploaded CSV schema and user question.
- Model outputs a `<SQL>...</SQL>` wrapped SQLite-compatible query.
- Safety checks to block destructive SQL (e.g., `DELETE`, `DROP`, `INSERT`).
- Executes the query in an in-memory SQLite DB and displays the result in Gradio.
- Designed for Colab GPU runtimes (but can be adapted to local machines with CUDA).

## Requirements
- Python 3.8+ (3.10+ recommended)
- CUDA-capable GPU for reasonable performance (optional but recommended)
- pip packages:
  - `transformers>=4.33.0`
  - `accelerate`
  - `bitsandbytes`
  - `safetensors`
  - `sentencepiece`
  - `gradio`
  - `huggingface_hub`
  - `torch` (CUDA build if using GPU)

## Quick start (Google Colab)
1. Open a Colab notebook and paste the code from `untitled9.py`.
2. Set runtime: `Runtime -> Change runtime type -> GPU`.
3. Install dependencies:
```bash
!pip install -q -U pip
!pip install -q "transformers>=4.33.0" accelerate bitsandbytes safetensors sentencepiece gradio
```
4. Provide Hugging Face token if required (use `huggingface_hub.login()` or set `HF_TOKEN` via environment variable).
5. Run the code. The Gradio UI will appear (the script uses `demo.launch(share=True)`).

## Quick start (Local)
1. Install dependencies in a terminal:
```bash
pip install -U pip
pip install "transformers>=4.33.0" accelerate bitsandbytes safetensors sentencepiece gradio
# Install torch appropriate to your CUDA version, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
2. Edit `untitled9.py` to remove notebook magic (if present) or run installs separately.
3. Set `HF_TOKEN` securely (environment variable recommended).
4. Run:
```bash
python untitled9.py
```

## How it works (high level)
- Loads a quantized Mistral instruct model and tokenizer.
- Builds a prompt describing the CSV schema and the user's NL question.
- Generates SQL wrapped in `<SQL>...</SQL>` tags.
- Extracts and checks SQL for safety (only read-only allowed).
- Loads the CSV into an in-memory SQLite table named `data`.
- Executes the SQL and returns results as a DataFrame shown in Gradio.

## Example
CSV:
```csv
id,name,total_sales
1,Alice,100.0
2,Bob,250.5
3,Carol,75.25
```
Question: `Show top 2 customers by total_sales`  
Model output (example): `<SQL>SELECT name, total_sales FROM data ORDER BY total_sales DESC LIMIT 2;</SQL>`

## Security & privacy notes
- Do **not** commit your Hugging Face token to public repos.
- The script restricts destructive SQL but isn't a full-proof guard — only run on safe/test data.
- When using `demo.launch(share=True)`, the UI becomes accessible via a public link.

## Troubleshooting
- Model download slow: use a fast connection and Colab GPU.
- OOM errors: use a smaller model, more VRAM, or CPU fallback (slow).
- `bitsandbytes` install issues: ensure compatibility with your CUDA & PyTorch versions.

## Suggested improvements
- Add model/tokenizer caching.
- Improve SQL sanitization and column whitelisting.
- Support multiple tables and JOIN generation.
- Provide a CPU-friendly fallback model.

---

If you want, I can also:
- Create a cleaned `run_local.py` script (no notebook magics).
- Produce a Colab-ready `.ipynb` notebook.

