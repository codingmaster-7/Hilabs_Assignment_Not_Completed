# README

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

## Preprocessing & validation-data logic

### What the current code does
- The pipeline reads an uploaded CSV into a `pandas.DataFrame` using `pd.read_csv(csv_file.name)`. The CSV is then loaded into an in-memory SQLite table named `data` before executing model-generated SQL.
- Basic schema/type inference is performed by `columns_to_schema_string(df)` which guesses each column's type using `pd.api.types.is_integer_dtype` and `is_float_dtype` and otherwise treats columns as `text`. That schema is embedded into the NL→SQL prompt so the model knows available columns and types.
- There is no explicit train/validation split or advanced preprocessing in the main script; uploaded CSV is used directly (after the simple type guessing) as the table `data` the model queries against.

### How validation data can be used (recommended)
If you have a separate validation CSV (or a verification notebook that produced validation splits), incorporate it to:
1. **Verify schema compatibility** — ensure the validation CSV has the same columns and compatible dtypes.
2. **Sanity-check model-generated SQL** — run the generated SQL against both the main dataset and the validation dataset to ensure results are consistent (for example: row counts, aggregate values, or sample rows).
3. **Apply deterministic preprocessing** — apply identical cleaning steps to both datasets (fill missing values, cast dtypes, normalize if needed) so SQL results are comparable.

### Minimal preprocessing helper (example)
Add this helper to your code to apply deterministic preprocessing to both train and validation DataFrames:

```python
def preprocess_df(df: pd.DataFrame, fill_numeric_with: float = 0.0, fill_text_with: str = "") -> pd.DataFrame:
    """Simple deterministic preprocessing to apply to both train and validation sets.
    - Ensures consistent dtypes for int/float/text using pandas type checks.
    - Fills NA values with defaults (configurable).
    - Removes duplicate columns and trims whitespace from column names.
    """
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Fill numeric columns
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_integer_dtype(ser) or pd.api.types.is_float_dtype(ser):
            df[col] = ser.fillna(fill_numeric_with)
            # Optional: coerce numeric columns to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_numeric_with)
            except Exception:
                pass
        else:
            df[col] = ser.fillna(fill_text_with).astype(str)
    return df
```

### Running generated SQL on validation data (example)
```python
# assume df is main data, val_df is optional validation data (both preprocessed)
conn_main = sqlite3.connect(":memory:")
df.to_sql("data", conn_main, index=False, if_exists="replace")
cur_main = conn_main.cursor()
cur_main.execute(sql)
main_rows = cur_main.fetchall()
main_cols = [d[0] for d in cur_main.description] if cur_main.description else []
result_main = pd.DataFrame(main_rows, columns=main_cols)
conn_main.close()

if val_df is not None:
    conn_val = sqlite3.connect(":memory:")
    val_df.to_sql("data", conn_val, index=False, if_exists="replace")
    cur_val = conn_val.cursor()
    cur_val.execute(sql)
    val_rows = cur_val.fetchall()
    val_cols = [d[0] for d in cur_val.description] if cur_val.description else []
    result_val = pd.DataFrame(val_rows, columns=val_cols)
    conn_val.close()
    # Basic consistency checks (examples)
    checks = {
        "main_row_count": len(result_main),
        "val_row_count": len(result_val),
        "matching_columns": result_main.columns.tolist() == result_val.columns.tolist()
    }
else:
    checks = {"main_row_count": len(result_main)}
```

### Notes & next steps
- The main script (`untitled9.py`) currently performs only simple type-guessing and no complex preprocessing or validation split. If you uploaded a separate verification notebook with preprocessing steps, paste or share the relevant cells and I will convert them into a clear README summary and, if you want, inject the exact code into `untitled9.py`.
- Consider adding a Gradio validation-file input and a small "Validate" button to run the checks above and show them in the UI.

---

If you want, I can:
- Add these helper functions directly into `untitled9.py` and update the Gradio UI to accept an optional validation CSV, or
- Extract and summarize the exact preprocessing steps from your verification notebook (please allow access or paste the relevant cells).

