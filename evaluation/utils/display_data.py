# build_results_table.py
import json
import pandas as pd
from pathlib import Path

# ---- EDIT THIS: map your method name -> json filepath ----
METHOD_FILES = {
    "Baseline":       "evaluation_baseline.json",
    "Expansion":      "evaluation_qoqa.json",
    "Abstraction": "evaluation_abstraction.json",
    "Disambiguation": "evaluation_disambiguation_2_0.json",
    "Decomposition":  "evaluation_decomposition.json",

}

RETRIEVERS = ["bm25", "dense", "hybrid"]   # keys in your JSON
METRICS_10 = [
    ("Recall", "Recall@10"),
    ("nDCG",   "NDCG@10"),
    ("MRR",    "MRR@10"),
]

def load_method_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_k10(method_name: str, data: dict) -> dict:
    """
    From a single-method JSON (your format), pull bm25/dense/hybrid at @10 for
    Recall, nDCG, MRR. Returns flat dict with keys like:
      'BM25 Recall@10', 'BM25 nDCG@10', 'BM25 MRR@10', 'Dense ...', 'Hybrid ...'
    """
    out = {}
    for retr in RETRIEVERS:
        if retr not in data:
            continue
        for group_key, metric_key in METRICS_10:
            try:
                val = data[retr][group_key][metric_key]
            except KeyError:
                val = None
            # Pretty column name with group
            nice_retr = retr.upper() if retr != "bm25" else "BM25"
            out[f"{nice_retr} {metric_key}"] = val
    out["_method"] = method_name
    return out

def build_combined_json(method_files: dict) -> dict:
    """
    Returns: {"Baseline": {bm25:{...}, dense:{...}, hybrid:{...}}, ...}
    """
    combined = {}
    for m, path in method_files.items():
        combined[m] = load_method_json(path)
    return combined

def build_excel_table(method_files: dict, out_excel: str = "results_k10.xlsx") -> pd.DataFrame:
    rows = []
    for m, path in method_files.items():
        data = load_method_json(path)
        row = extract_k10(m, data)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("_method").rename_axis("Method")

    # Reorder columns into grouped order
    ordered_cols = []
    for retr in ["BM25", "DENSE", "HYBRID"]:
        for _, metric_key in METRICS_10:
            ordered_cols.append(f"{retr} {metric_key}")
    # some keys might be missing if a retriever wasn't run; keep those that exist
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    # Optional: prettier multiindex columns for grouping in Excel
    tuples = []
    for c in df.columns:
        retr, metric = c.split(" ", 1)
        # Normalize names
        retr_pretty = {"BM25":"BM25", "DENSE":"Dense", "HYBRID":"Hybrid"}[retr]
        metric_pretty = metric.replace("@10","@10")
        # Convert names like "NDCG@10" to "nDCG@10" if you prefer
        metric_pretty = metric_pretty.replace("NDCG", "nDCG")
        tuples.append((retr_pretty, metric_pretty))
    df.columns = pd.MultiIndex.from_tuples(tuples, names=["Retriever", "Metric"])

    # Write Excel with a frozen header row
    with pd.ExcelWriter(out_excel, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="K10")
        ws = xw.sheets["K10"]
        ws.freeze_panes(1, 1)
        # Optional: set column widths
        for i, _ in enumerate(df.columns, start=2):
            ws.set_column(i, i, 14)

    print(f"Wrote: {out_excel}")
    return df

if __name__ == "__main__":
    # 1) combined JSON (optional, if you want a single file with everything)
    combined = build_combined_json(METHOD_FILES)
    Path("combined_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("Wrote: combined_results.json")

    # 2) Excel table (rows=methods; grouped columns=(BM25,Dense,Hybrid) × metrics @10)
    build_excel_table(METHOD_FILES, out_excel="results_k10.xlsx")
