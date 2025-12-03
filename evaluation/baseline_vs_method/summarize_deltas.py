import json

def summarize_deltas(delta_json_path):
    """
    Reads a per-query delta JSON of the form:
        {
          "qid1": {
            "bm25": {
              "baseline": {...},
              "method": {...},
              "delta": {"recall": dR, "ndcg": dN, "mrr": dM}
            },
            "dense": {...},
            "hybrid": {...}
          },
          ...
        }

    For each retriever (bm25, dense, hybrid) and each metric (recall, ndcg, mrr),
    counts how many queries:
      - improved (delta > 0)
      - worsened (delta < 0)
      - stayed the same (delta == 0)
    and prints counts + percentages.
    """
    with open(delta_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retrievers = ["bm25", "dense", "hybrid"]
    metrics = ["recall", "ndcg", "mrr"]

    stats = {
        r: {
            m: {"improved": 0, "worsened": 0, "same": 0}
            for m in metrics
        }
        for r in retrievers
    }

    num_queries = len(data)

    for qid, per_ret in data.items():
        for r in retrievers:
            if r not in per_ret:
                continue
            delta = per_ret[r].get("delta", {})
            for m in metrics:
                d = delta.get(m, 0.0)
                if d > 0:
                    stats[r][m]["improved"] += 1
                elif d < 0:
                    stats[r][m]["worsened"] += 1
                else:
                    stats[r][m]["same"] += 1

    print(f"Total queries: {num_queries}")
    for r in retrievers:
        print(f"\nRetriever: {r.upper()}")
        for m in metrics:
            c = stats[r][m]
            imp = c["improved"]
            wor = c["worsened"]
            same = c["same"]

            imp_pct = 100.0 * imp / num_queries
            wor_pct = 100.0 * wor / num_queries
            same_pct = 100.0 * same / num_queries

            print(f"  Metric: {m}")
            print(f"    improved : {imp:4d} ({imp_pct:5.1f} %)")
            print(f"    worsened : {wor:4d} ({wor_pct:5.1f} %)")
            print(f"    same     : {same:4d} ({same_pct:5.1f} %)")


if __name__ == "__main__":
    summarize_deltas("per_query_deltas_disambiguation.json")
