import os
import json
import time
from tqdm import tqdm

from retrieval import load_document_store, create_pipeline, retrieve_docs
from haystack.document_stores import ElasticsearchDocumentStore
from abstract import generate_abstracted_query  # your function

ES_INDEX_NAME = "bm25_docs"
ES_HOST = "localhost"
ES_PORT = 9200


def _merge_dedup_select_5_5_strict(orig_list, abs_list, top_k=10, per_side=5):
    """
    Inputs (already ranked lists from retrieve_docs):
      - orig_list: top_k from original query (same retriever)
      - abs_list : top_k from abstracted query (same retriever)

    Steps:
      1) Merge lists (≈20 items).
      2) Cross-deduplicate by doc_id:
         - keep higher score;
         - if scores tie, keep the one that appeared earlier in its OWN source list;
         - if rank also ties (rare), prefer 'orig'.
      3) From the deduped winners:
         - take up to 5 from 'orig' (preserving original order),
         - take up to 5 from 'abs'  (preserving original order).
      4) If fewer than top_k selected, fill from whichever side still has leftovers
         (preserving that side’s order). If both sides short, alternate.
    """
    # Build rank maps (lower index = better rank in its source)
    orig_rank = {d.get("doc_id"): i for i, d in enumerate(orig_list or []) if d.get("doc_id")}
    abs_rank  = {d.get("doc_id"): i for i, d in enumerate(abs_list  or []) if d.get("doc_id")}

    # Cross-deduplicate with provenance + rank
    best = {}  # doc_id -> {item..., "_src": "orig"/"abs", "_rank": int}
    def consider(lst, src, rmap):
        for d in lst or []:
            did = d.get("doc_id")
            if not did:
                continue
            s = d.get("score", float("-inf"))
            r = rmap.get(did, 10**9)
            prev = best.get(did)
            if prev is None:
                dd = dict(d); dd["_src"] = src; dd["_rank"] = r
                best[did] = dd
            else:
                ps, pr, psrc = prev.get("score", float("-inf")), prev.get("_rank", 10**9), prev.get("_src")
                if s > ps:
                    dd = dict(d); dd["_src"] = src; dd["_rank"] = r
                    best[did] = dd
                elif s == ps:
                    # tie → better rank wins; if still tie, prefer orig
                    if r < pr or (r == pr and src == "orig" and psrc != "orig"):
                        dd = dict(d); dd["_src"] = src; dd["_rank"] = r
                        best[did] = dd

    consider(orig_list, "orig", orig_rank)
    consider(abs_list,  "abs",  abs_rank)

    # Split by provenance, preserve each side’s original order via stored rank
    orig_win = [d for d in best.values() if d.get("_src") == "orig"]
    abs_win  = [d for d in best.values() if d.get("_src") == "abs"]
    orig_win.sort(key=lambda x: x.get("_rank", 10**9))
    abs_win.sort(key=lambda x: x.get("_rank", 10**9))

    # Take up to 5 from each
    sel_orig = orig_win[:min(per_side, len(orig_win))]
    sel_abs  = abs_win [:min(per_side, len(abs_win))]
    selected = sel_orig + sel_abs

    # Fill if needed
    need = top_k - len(selected)
    if need > 0:
        rem_orig = orig_win[len(sel_orig):]
        rem_abs  = abs_win[len(sel_abs):]

        # If one side didn’t reach per_side, fill from the OTHER side first
        if len(sel_orig) < per_side and rem_abs:
            take = min(need, len(rem_abs)); selected.extend(rem_abs[:take]); need -= take
        if need > 0 and len(sel_abs) < per_side and rem_orig:
            take = min(need, len(rem_orig)); selected.extend(rem_orig[:take]); need -= take

        # If still need, alternate remaining
        i_o = i_a = 0
        while need > 0 and (i_o < len(rem_orig) or i_a < len(rem_abs)):
            if i_o < len(rem_orig):
                selected.append(rem_orig[i_o]); i_o += 1; need -= 1
                if need == 0: break
            if i_a < len(rem_abs):
                selected.append(rem_abs[i_a]); i_a += 1; need -= 1

    # Cleanup helper fields
    for d in selected:
        d.pop("_src", None); d.pop("_rank", None)

    return selected  # may be < top_k only if total uniques across both sides < top_k


def run_retrieval_iteratively_abstraction(
    queries_path="datasets/nq/queries.jsonl",
    output_path_abstraction="retrieval_results_abstraction.jsonl",
    output_path_original="retrieval_results_original.jsonl",
    start_from=1,
    n_queries=300,
    top_k=10,
    target_each=5,  # desired per-side count (5/5)
):
    """
    For each query:
      - Generate abstracted query.
      - Retrieve exactly top_k=10 for BOTH original and abstracted.
      - Merge the results of both methods, deduplicate and keep the 5 highest scoring from original and 5 abstracted
    """
    print("🔹 Loading document stores...")
    store = load_document_store()
    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME)

    print("🔹 Initializing retrieval pipeline...")
    pipe = create_pipeline(store, es_store)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    if n_queries:
        queries = queries[start_from:start_from + n_queries]
    else:
        queries = queries[start_from:]

    print(f"🚀 Running {len(queries)} queries starting from index {start_from}")

    retrievers = ["bm25", "dense", "hybrid"]

    with open(output_path_abstraction, "a", encoding="utf-8") as f_abs, \
         open(output_path_original, "a", encoding="utf-8") as f_org:

        for i, q in enumerate(tqdm(queries, desc="Retrieving (abstraction + original-only)")):
            qid = q.get("_id", f"q{i + start_from}")
            question = q.get("text")

            try:
                t0 = time.perf_counter()

                # Abstract the query
                try:
                    abs_query = generate_abstracted_query(question)
                except Exception as e:
                    print("❌ Abstraction failed:", e)
                    raise
                if not isinstance(abs_query, str):
                    abs_query = str(abs_query)

                # Retrieve exactly top_k for both
                try:
                    orig_ret = retrieve_docs(pipe, question, top_k=top_k)
                except Exception as e:
                    print("❌ Abstraction failed:", e)
                    raise
                # Baseline original-only (dedup within each retriever's list)
                baseline_record = {
                    "query_id": qid,
                    "query": question,
                    "bm25": orig_ret["bm25"],
                    "dense": orig_ret["dense"],
                    "hybrid": orig_ret["hybrid"],
                    "runtime_sec": round(time.perf_counter() - t0, 4),
                }
                f_org.write(json.dumps(baseline_record, ensure_ascii=False) + "\n");
                f_org.flush()
                abs_ret  = retrieve_docs(pipe, abs_query, top_k=top_k)
                # Abstraction merge with 5/5 preference and fill
                final = {}
                for r in retrievers:
                    final[r] = _merge_dedup_select_5_5_strict(
                        orig_list=orig_ret.get(r, []),
                        abs_list=abs_ret.get(r, []),
                        top_k=10,
                        per_side=5
                    )

                abstraction_record = {
                    "query_id": qid,
                    "query": question,
                    "abstracted_query": abs_query,
                    "bm25": final["bm25"],
                    "dense": final["dense"],
                    "hybrid": final["hybrid"],
                    "runtime_sec": round(time.perf_counter() - t0, 4)
                }
                f_abs.write(json.dumps(abstraction_record, ensure_ascii=False) + "\n"); f_abs.flush()

            except Exception as e:
                raise
    print(f"✅ Done. Abstraction: {output_path_abstraction} | Original-only: {output_path_original}")


if __name__ == "__main__":
    run_retrieval_iteratively_abstraction(
        queries_path="datasets/nq/queries.jsonl",
        output_path_abstraction="retrieval_results_abstraction_parallel.jsonl",
        output_path_original="retrieval_results_original_parallel.jsonl",
        start_from=369,
        n_queries=1,
        top_k=10,
        target_each=5,
    )
    run_retrieval_iteratively_abstraction(
        queries_path="datasets/nq/queries.jsonl",
        output_path_abstraction="retrieval_results_abstraction_parallel.jsonl",
        output_path_original="retrieval_results_original_parallel.jsonl",
        start_from=384,
        n_queries=1,
        top_k=10,
        target_each=5,
    )
    run_retrieval_iteratively_abstraction(
        queries_path="datasets/nq/queries.jsonl",
        output_path_abstraction="retrieval_results_abstraction_parallel.jsonl",
        output_path_original="retrieval_results_original_parallel.jsonl",
        start_from=184,
        n_queries=1,
        top_k=10,
        target_each=5,
    )
    run_retrieval_iteratively_abstraction(
        queries_path="datasets/nq/queries.jsonl",
        output_path_abstraction="retrieval_results_abstraction_parallel.jsonl",
        output_path_original="retrieval_results_original_parallel.jsonl",
        start_from=500,
        n_queries=1,
        top_k=10,
        target_each=5,
    )
