import os
import json
import time
from tqdm import tqdm
from haystack.document_stores import ElasticsearchDocumentStore

from src.retrieval import load_document_store, create_pipeline, retrieve_docs
from qoqa import get_optimized_query_with_qoqa

ES_INDEX_NAME = "bm25_docs"
ES_HOST = "localhost"
ES_PORT = 9200


def run_retrieval_iteratively_qoqa(
    queries_path="datasets/nq/queries.jsonl",
    output_path="retrieval_results_qoqa.jsonl",
    start_from=0,
    n_queries=None,
    top_k=10
):
    """
    For each question:
      1) Generate ONE optimized query via QOQA (iterative by default).
      2) Retrieve top_k per retriever (bm25/dense/hybrid) using ONLY the optimized query.
      3) Save JSONL with: query_id, query (original), optimized_query, bm25, dense, hybrid, runtime_sec.

    Notes:
      - Assumes `qoqa_iterative_best_query(question, pipe, ...)` is available and returns:
          {"best_query": <str>, "score": <float>, "candidates": [...]}
      - `retrieve_docs(pipe, q, top_k)` must return dict with keys {"bm25","dense","hybrid"},
        each a list of items like {"doc_id": ..., "score": float, "text": str?}.
    """
    print("🔹 Loading document stores...")
    store = load_document_store()
    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME)

    print("🔹 Initializing retrieval pipeline...")
    pipe = create_pipeline(store, es_store)

    # --- Load queries ---
    with open(queries_path, "r", encoding="utf-8") as f:
        all_qs = [json.loads(line) for line in f]

    if n_queries is not None:
        queries = all_qs[start_from:start_from + n_queries]
    else:
        queries = all_qs[start_from:]

    print(f"🚀 Running {len(queries)} queries starting from index {start_from}")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, q in enumerate(tqdm(queries, desc="Retrieving (QOQA)")):
            qid = q.get("_id", f"q{i + start_from}")
            question = q.get("text")

            t0 = time.perf_counter()
            try:
                optimized_q = get_optimized_query_with_qoqa(
                    question=question, pipe=pipe
                )

                retrieved = retrieve_docs(pipe, optimized_q, top_k=top_k)

                # 3) Record
                record = {
                    "query_id": qid,
                    "query": question,
                    "expanded_query": optimized_q,
                    "bm25": retrieved.get("bm25", [])[:top_k],
                    "dense": retrieved.get("dense", [])[:top_k],
                    "hybrid": retrieved.get("hybrid", [])[:top_k],
                    "runtime_sec": round(time.perf_counter() - t0, 4),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                raise

    print(f"✅ Finished {len(queries)} queries. Results saved to: {output_path}")


if __name__ == "__main__":
    run_retrieval_iteratively_qoqa(
        queries_path="datasets/nq/queries.jsonl",
        output_path="retrieval_results_qoqa.jsonl",
        start_from=276,
        n_queries=2,
        top_k=10
    )
