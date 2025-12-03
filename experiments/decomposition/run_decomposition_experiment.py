import os

from src.retrieval import load_document_store, create_pipeline, retrieve_docs
from haystack.document_stores import ElasticsearchDocumentStore

from decompose import generate_decomposed_query

INDEX_PATH = "./faiss_index"
FAISS_FILE = os.path.join(INDEX_PATH, "faiss_index")

ES_INDEX_NAME = "bm25_docs"
ES_HOST = "localhost"
ES_PORT = 9200

import time
from tqdm import tqdm
#TODO: BETTER HANDLING OF CASE IF LLM RETURNS 1 QUERY ONLY
# --- Function each process runs ---
import json

def run_retrieval_iteratively(
    queries_path="datasets/nq/queries.jsonl",
    output_path="retrieval_results_decomposition.jsonl",
    start_from=173,
    n_queries=1,
    top_k=10,
):
    """
    Runs the retrieval pipeline iteratively and saves results progressively.
    You can control how many queries to run and resume later.
    """
    #add time so we compare how long they took
    # --- Load document stores ---
    print("🔹 Loading document stores...")
    store = load_document_store()
    es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME,
                                          )

    # --- Create pipeline once ---
    print("🔹 Initializing retrieval pipeline...")
    pipe = create_pipeline(store, es_store)

    # --- Load queries ---
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]

    if n_queries:
        queries = queries[start_from:start_from + n_queries]
    else:
        queries = queries[start_from:]

    print(f"🚀 Running {len(queries)} queries starting from index {start_from}")

    # --- Run retrieval iteratively ---
    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, q in enumerate(tqdm(queries, desc="Retrieving")):
            qid = q.get("_id", f"q{i + start_from}")
            question = q.get("text")

            t0 = time.perf_counter()
            try:
                sub_queries = generate_decomposed_query(question)
                if not isinstance(sub_queries, list):
                    sub_queries = [sub_queries]


                retrievers = ["bm25", "dense", "hybrid"]
                collected = {r: [] for r in retrievers}

                # STEP 2 — Retrieve per sub-query and take top 5 directly
                for sq in sub_queries:
                    retrieved = retrieve_docs(pipe, sq, top_k=top_k)

                    for r in retrievers:
                        # Take top 5 per sub-query
                        top_docs = sorted(retrieved[r], key=lambda d: d["score"], reverse=True)[:5]
                        collected[r].extend(top_docs)

                # STEP 3 — Merge across all sub-queries and ensure always top_k docs
                final = {}
                for r in retrievers:
                    merged = {}
                    for doc in collected[r]:
                        doc_id = doc["doc_id"]
                        score = doc["score"]
                        if doc_id not in merged or score > merged[doc_id]["score"]:
                            merged[doc_id] = doc

                    # Sort by score
                    ranked = sorted(merged.values(), key=lambda d: d["score"], reverse=True)

                    # If we have fewer than top_k docs, pad with lowest-score doc copies
                    if len(ranked) < top_k and ranked:
                        last_doc = ranked[-1]
                        ranked += [last_doc.copy() for _ in range(top_k - len(ranked))]

                    # Truncate to top_k exactly
                    final[r] = ranked[:top_k]

                # STEP 4 — Store everything
                result = {
                    "query_id": qid,
                    "query": question,
                    "sub_queries": sub_queries,
                    "bm25": final["bm25"],
                    "dense": final["dense"],
                    "hybrid": final["hybrid"],
                    "runtime_sec": round(time.perf_counter() - t0, 4),
                }

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()  # write to disk immediately
            except Exception as e:
                raise

    print(f"✅ Finished {len(queries)} queries. Results saved to: {output_path}")





if __name__ == "__main__":
    #TODO: test the thread on 5 queries see how fast they are/ if its faster/ check how results.jsonl looks like if its the wanted format
    #DO A TEST with decomposition see what llms answers and how to do it for
    run_retrieval_iteratively()

