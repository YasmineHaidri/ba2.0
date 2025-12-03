def merge_dedup_disambiguation(subquery_lists, top_k=10):
   """
   Fusion heuristic for Disambiguation (score-based, with per-sub-query quota and
   fallback using leftover docs):

   - subquery_lists: list of lists, one per disambiguated sub-query.
     Each inner list is the retrieval result for that sub-query and one retriever:
       [ { "doc_id": ..., "score": ... }, ... ]

   Steps:
     1) Compute a quota per sub-query that sums to top_k
        (e.g. top_k=10, 3 sub-queries -> [4,3,3]).
     2) For each sub-query i:
          - sort its docs by score (descending),
          - split into:
              * primary_i   = first quota[i] docs
              * leftover_i  = remaining docs for that sub-query
     3) Pool all primary_i, deduplicate by doc_id (keep highest score),
        and sort by score.
     4) If there are still fewer than top_k unique docs:
          - pool all leftover docs across sub-queries,
          - sort leftovers by score,
          - iterate through them in descending score:
              * for each doc_id not yet in the merged set, add it,
              * stop when we reach top_k unique docs or run out of leftovers.
     5) Sort the final merged set by score (descending) and truncate to top_k.

   Returns:
     List of at most top_k document dicts (no padding, no duplicate doc_ids).
   """
   num_sides = len(subquery_lists)

   # 1) per-sub-query quotas that sum to top_k
   base = top_k // num_sides
   extra = top_k % num_sides
   quotas = [base + (1 if i < extra else 0) for i in range(num_sides)]

   # 2) primary candidates and leftovers per sub-query
   primary_candidates = []
   leftovers_all = []

   for i, lst in enumerate(subquery_lists):

       q = quotas[i]

       # sort docs from this sub-query by score (descending)
       sorted_lst = sorted(
           lst,
           key=lambda d: d.get("score", 0.0),
           reverse=True,
       )

       primary = sorted_lst[:q]
       leftover = sorted_lst[q:]

       primary_candidates.extend(primary)
       leftovers_all.extend(leftover)

   # 3) deduplicate primary candidates by doc_id, keep highest score
   merged = {}  # doc_id -> best doc
   for doc in primary_candidates:
       did = doc.get("doc_id")
       s = doc.get("score", 0.0)
       prev = merged.get(did)
       if prev is None or s > prev.get("score", 0.0):
           merged[did] = doc

   # rank primary merged docs
   ranked = sorted(
       merged.values(),
       key=lambda d: d.get("score", 0.0),
       reverse=True,
   )

   # 4) if we still have fewer than top_k unique docs, fill from leftovers
   if len(ranked) < top_k and leftovers_all:
       # sort all leftovers by score (descending)
       leftovers_sorted = sorted(
           leftovers_all,
           key=lambda d: d.get("score", 0.0),
           reverse=True,
       )

       for doc in leftovers_sorted:
           if len(merged) >= top_k:
               break
           did = doc.get("doc_id")
           if did in merged:
               # already have this doc_id from primary phase
               continue
           merged[did] = doc

       # re-rank after adding leftovers
       ranked = sorted(
           merged.values(),
           key=lambda d: d.get("score", 0.0),
           reverse=True,
       )

   # 5) final top_k (or fewer if total unique docs < top_k)
   return ranked[:top_k]

import json
import time
from tqdm import tqdm
from haystack.document_stores import ElasticsearchDocumentStore

from src.retrieval import load_document_store, create_pipeline, retrieve_docs

ES_INDEX_NAME = "bm25_docs"
ES_HOST = "localhost"
ES_PORT = 9200


def run_retrieval_iteratively_disambiguation(
   disamb_input_path="retrieval_results_disambiguation.jsonl",
   output_path_disambiguation="retrieval_results_disambiguation_rerun.jsonl",
   start_from=0,
   n_queries=None,
   top_k=10,
):
   """
   Re-run Disambiguation retrieval using *existing* disambiguated queries.

   For each line in `disamb_input_path`:
     - read:
         {
           "query_id": ...,
           "query": ...,
           "disambiguated_queries": [...]
         }
     - use `disambiguated_queries` as sub-queries (no LLM call),
     - retrieve top_k docs for each sub-query (bm25, dense, hybrid),
     - fuse per retriever with `merge_dedup_disambiguation`,
     - write new results to `output_path_disambiguation`.

   If for any (query, retriever) the fused list is empty,
   the function raises an exception (fail-fast).
   """

   # --- load precomputed disambiguated queries ---
   with open(disamb_input_path, "r", encoding="utf-8") as f:
       records = [json.loads(line) for line in f]

   if n_queries is not None:
       records = records[start_from:start_from + n_queries]
   else:
       records = records[start_from:]

   print(f"🔹 Loaded {len(records)} disambiguated records from {disamb_input_path}")
   print("🔹 Loading document stores...")
   store = load_document_store()
   es_store = ElasticsearchDocumentStore(host=ES_HOST, port=ES_PORT, index=ES_INDEX_NAME)

   print("🔹 Initializing retrieval pipeline...")
   pipe = create_pipeline(store, es_store)

   retrievers = ["bm25", "dense", "hybrid"]

   # Open fresh output file
   with open(output_path_disambiguation, "a", encoding="utf-8") as f_dis:

       for rec in tqdm(records, desc="Retrieving (disambiguation rerun)"):
           qid = rec.get("query_id")
           question = rec.get("query")
           dis_queries = rec.get("disambiguated_queries", [])

           # normalize and drop empties
           if not isinstance(dis_queries, list):
               dis_queries = [dis_queries]
           dis_queries = [dq for dq in dis_queries if dq]

           if not dis_queries:
               raise ValueError(f"No disambiguated queries for {qid}: {question}")

           t0 = time.perf_counter()

           # retrieve for each existing disambiguated sub-query
           per_retriever_lists = {r: [] for r in retrievers}

           for dq in dis_queries:
               sub_ret = retrieve_docs(pipe, dq, top_k=top_k)
               for r in retrievers:
                   docs = sub_ret.get(r, [])
                   # we allow short lists, but not None
                   if docs is None:
                       raise RuntimeError(
                           f"Retriever '{r}' returned None for query_id={qid}, sub_query='{dq}'"
                       )
                   per_retriever_lists[r].append(docs)

           # fuse across sub-queries with your new fusion
           final = {}
           for r in retrievers:
               fused = merge_dedup_disambiguation(
                   subquery_lists=per_retriever_lists[r],
                   top_k=top_k,
               )
               # FAIL FAST if the fused list is empty for this retriever
               if not fused:
                   raise RuntimeError(
                       f"Empty fused list for retriever '{r}' "
                       f"on query_id={qid}, dis_queries={dis_queries}"
                   )
               final[r] = fused

           dis_record = {
               "query_id": qid,
               "query": question,
               "disambiguated_queries": dis_queries,
               "bm25": final["bm25"],
               "dense": final["dense"],
               "hybrid": final["hybrid"],
               "runtime_sec": round(time.perf_counter() - t0, 4),
           }
           f_dis.write(json.dumps(dis_record, ensure_ascii=False) + "\n")
           f_dis.flush()

   print(f"✅ Done. Disambiguation results written to: {output_path_disambiguation}")


if __name__ == "__main__":
  #TODO: test the thread on 5 queries see how fast they are/ if its faster/ check how results.jsonl looks like if its the wanted format
  #DO A TEST with decomposition see what llms answers and how to do it for
  run_retrieval_iteratively_disambiguation(
      disamb_input_path="retrieval_results_disambiguation.jsonl",
      output_path_disambiguation="retrieval_results_disambiguation_2_0.jsonl",
      start_from=501,
      n_queries=1,
      top_k=10,
  )
