import json
import math

def load_qrels_tsv(qrels_path):
    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            # Skip empty lines or header
            if len(parts) != 3 or parts[0].lower() in {"query-id", "qid"}:
                continue
            qid, docid, score = parts
            qrels.setdefault(qid, {})[docid] = int(score)
    return qrels


def load_results_json(path):
    """
    Expects BEIR-style results:
        {
            "qid1": {"doc1": score, "doc2": score, ...},
            "qid2": ...
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# 3. Metrics per query
# --------------------------

def compute_metrics_for_query(ranked_doc_ids, qrels_for_q, k=10):
    """
    Compute Recall@k, nDCG@k, MRR@k for a single query.
    ranked_doc_ids: ordered list of doc_ids (best first)
    qrels_for_q: {doc_id: relevance} for this query
    """
    if qrels_for_q is None:
        qrels_for_q = {}

    ranked_doc_ids = ranked_doc_ids[:k]

    # total relevant in qrels
    total_rel = sum(1 for rel in qrels_for_q.values() if rel > 0)
    if total_rel == 0:
        return 0.0, 0.0, 0.0

    # Recall@k
    rel_in_topk = sum(1 for d in ranked_doc_ids if qrels_for_q.get(d, 0) > 0)
    recall = rel_in_topk / total_rel

    # nDCG@k (binary relevance)
    dcg = 0.0
    for i, d in enumerate(ranked_doc_ids):
        rel = qrels_for_q.get(d, 0)
        if rel > 0:
            dcg += 1.0 / math.log2(i + 2)  # rank = i+1

    ideal_rels = [1] * min(total_rel, k)
    idcg = 0.0
    for i, _ in enumerate(ideal_rels):
        idcg += 1.0 / math.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MRR@k
    mrr = 0.0
    for i, d in enumerate(ranked_doc_ids):
        if qrels_for_q.get(d, 0) > 0:
            mrr = 1.0 / (i + 1)
            break

    return recall, ndcg, mrr


def compute_per_query_metrics(results, qrels, k=10):
    """
    results: {qid: {doc_id: score, ...}}
    qrels:  {qid: {doc_id: rel, ...}}
    returns: {qid: {"recall": R, "ndcg": N, "mrr": M}}
    """
    per_query = {}
    for qid, doc_scores in results.items():
        ranked_doc_ids = sorted(
            doc_scores.keys(),
            key=lambda d: doc_scores[d],
            reverse=True
        )
        qrels_for_q = qrels.get(qid, {})
        recall, ndcg, mrr = compute_metrics_for_query(ranked_doc_ids, qrels_for_q, k=k)
        per_query[qid] = {"recall": recall, "ndcg": ndcg, "mrr": mrr}
    return per_query


# --------------------------
# 4. Compare method vs baseline and write JSON
# --------------------------

def per_query_deltas_vs_baseline(
    baseline_prefix,
    method_prefix,
    qrels_path,
    k=10,
    output_json="per_query_deltas.json",
):
    """
    Computes per-query Recall@k, nDCG@k, MRR@k for baseline and a given method
    (e.g. Expansion) across bm25, dense, hybrid, and writes a JSON with:
        {
          "qid1": {
            "bm25": {
              "baseline": {...},
              "method": {...},
              "delta": {...}
            },
            "dense": {...},
            "hybrid": {...}
          },
          ...
        }
    """

    retrievers = ["bm25", "dense", "hybrid"]
    qrels = load_qrels_tsv(qrels_path)

    base_metrics = {}
    meth_metrics = {}

    # compute per-query metrics for each retriever
    for r in retrievers:
        base_results = load_results_json(f"{baseline_prefix}_{r}.json")
        meth_results = load_results_json(f"{method_prefix}_{r}.json")

        base_metrics[r] = compute_per_query_metrics(base_results, qrels, k=k)
        meth_metrics[r] = compute_per_query_metrics(meth_results, qrels, k=k)

    # collect all query_ids
    all_qids = set()
    for r in retrievers:
        all_qids.update(base_metrics[r].keys())

    out = {}

    for qid in all_qids:
        out[qid] = {}
        for r in retrievers:
            base_vals = base_metrics[r].get(qid, {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0})
            meth_vals = meth_metrics[r].get(qid, {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0})

            out[qid][r] = {
                "baseline": base_vals,
                "method": meth_vals,
                "delta": {
                    "recall": meth_vals["recall"] - base_vals["recall"],
                    "ndcg": meth_vals["ndcg"] - base_vals["ndcg"],
                    "mrr": meth_vals["mrr"] - base_vals["mrr"],
                },
            }

    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(out, f_out, indent=2, ensure_ascii=False)

    print(f"Per-query deltas written to {output_json}")


if __name__ == "__main__":
    # Example usage for Expansion:
    # assumes files like:
    #   baseline_bm25.json, baseline_dense.json, baseline_hybrid.json
    #   expansion_bm25.json, expansion_dense.json, expansion_hybrid.json
    per_query_deltas_vs_baseline(
        baseline_prefix="ready_eva/beir_results_baseline",
        method_prefix="ready_eva/beir_results_disambiguation_2_0",
        qrels_path="../../datasets/nq/qrels/test.tsv",
        k=10,
        output_json="per_query_deltas_disambiguation.json",
    )
