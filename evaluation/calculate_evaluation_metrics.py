from beir.retrieval.evaluation import EvaluateRetrieval
import pandas as pd
import json

def convert_to_beir_format_jsonl(input_jsonl, output_prefix):
    """
    Convert a JSONL retrieval output (one JSON object per line)
    into BEIR-compatible JSONs.
    Creates:
        beir_results_bm25.json,
        beir_results_dense.json,
        beir_results_hybrid.json
    """

    retrievers = ["bm25", "dense", "hybrid"]
    results = {r: {} for r in retrievers}

    data = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    # --- Convert entries to BEIR format (same logic as before) ---
    for entry in data:
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("query_id", "unknown"))
        for retriever in retrievers:
            if retriever in entry and isinstance(entry[retriever], list):
                results[retriever][qid] = {
                    str(d["doc_id"]): float(d["score"])
                    for d in entry[retriever]
                    if isinstance(d, dict) and "doc_id" in d and "score" in d
                }

    # --- Write BEIR-format outputs ---
    for retriever in retrievers:
        out_path = f"{output_prefix}_{retriever}.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(results[retriever], f_out, indent=2, ensure_ascii=False)
        print(f"✅ Saved BEIR-format file for {retriever}: {out_path}")


def evaluate_with_beir_metrics(results_prefix, qrels_path, output_json:str, ks=[1,3,5,10]):
    """
    Evaluate bm25, dense, and hybrid retrievers using Recall@K, MRR, and nDCG.
    Saves metrics to a JSON file.
    """
    retrievers = ["bm25", "dense", "hybrid"]
    evaluator = EvaluateRetrieval()


    qrels_nq = load_qrels_tsv(qrels_path)

    all_metrics = {}

    # --- Evaluate each retriever ---
    for retriever in retrievers:
        result_path = f"{results_prefix}_{retriever}.json"
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        qids_with_results = set(results.keys())
        qrels = {qid: rels for qid, rels in qrels_nq.items() if qid in qids_with_results}

        print(f"\n🔍 Evaluating {retriever.upper()}")
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=ks)
        mrr = evaluator.evaluate_custom(qrels, results, metric="mrr", k_values=ks)

        all_metrics[retriever] = {
            "nDCG": ndcg,
            "Recall": recall,
            "MRR": mrr
        }

    # --- Save results to JSON ---
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(all_metrics, f_out, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation metrics saved to: {output_json}")
    return all_metrics

def load_qrels_tsv(qrels_path):
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3 or parts[0].lower() in {"query-id", "qid"}:
                continue
            qid, docid, score = parts
            qrels.setdefault(qid, {})[docid] = int(score)
    return qrels


if __name__ == "__main__":
    #test_metrics('beir_results_original', qrels_path="../datasets/nq/qrels/test.tsv")
    convert_to_beir_format_jsonl('retrieval_results_qoqa_500.jsonl', "beir_results_qoqa")
    evaluate_with_beir_metrics('beir_results_qoqa',qrels_path="../datasets/nq/qrels/test.tsv",output_json="evaluation_qoqa.json")
