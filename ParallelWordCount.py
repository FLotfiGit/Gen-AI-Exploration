### parallel work count 
# sample code for parallelization of large file process
# Sep 2025

from multiprocessing import Pool
from collections import Counter

def process_chunk(file_chunk):
    counts = Counter()
    for line in file_chunk:
        words = line.strip().split()
        counts.update(words)
    return counts

def merge_counts(results):
    total = Counter()
    for r in results:
        total.update(r)
    return total

if __name__ == "__main__":
    # Split file into chunks
    with open("bigfile.txt", "r") as f:
        lines = f.readlines()
    chunk_size = len(lines) // 4   # 4 parallel workers
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]

    with Pool(4) as p:
        results = p.map(process_chunk, chunks)

    final_counts = merge_counts(results)
    print(final_counts.most_common(10))
