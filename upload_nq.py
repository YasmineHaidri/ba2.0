import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from rag_pipeline.index_documents import DATA_FOLDER

BEIR_OUT = "./datasets"
DATASET = "nq"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(BEIR_OUT, exist_ok=True)

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
target = os.path.join(BEIR_OUT, "datasets_1/nq")
if not os.path.exists(target):
    print("Downloading BEIR NQ (this may take a minute)...")
    util.download_and_unzip(url, BEIR_OUT)
else:
    print("BEIR NQ already downloaded.")

data_folder = os.path.join(BEIR_OUT, DATASET)
print("Loading BEIR NQ from:", data_folder)
corpus, queries, qrels = GenericDataLoader(data_folder).load(split="test")

print(f"Corpus passages: {len(corpus)}, queries: {len(queries)}, qrels: {len(qrels)}")

