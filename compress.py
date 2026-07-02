import gzip
import shutil

with open("data/candidates.jsonl", "rb") as fin:
    with gzip.open("data/candidates.jsonl.gz", "wb") as fout:
        shutil.copyfileobj(fin, fout)