import math
from collections import Counter, defaultdict

def tokenize(s):
    return [w.lower() for w in s.split()]  # replace with a better tokenizer if needed

def build_tfidf(corpus):
    # corpus: list[str]
    docs = [tokenize(doc) for doc in corpus]
    N = len(docs)

    # document frequency
    df = defaultdict(int)
    for doc in docs:
        for t in set(doc):
            df[t] += 1

    # vocabulary index
    vocab = {t:i for i, t in enumerate(sorted(df.keys()))}

    # idf
    idf = {t: math.log((1+N)/(1+df[t])) + 1 for t in vocab}

    # tf-idf vectors (L2-normalized)
    vectors = []
    for doc in docs:
        counts = Counter(doc)
        L = len(doc) if len(doc) > 0 else 1
        vec = [0.0]*len(vocab)
        # compute tf-idf
        for t, c in counts.items():
            if t in vocab:
                i = vocab[t]
                tf = c / L
                vec[i] = tf * idf[t]
        # L2 normalize (optional but common)
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        vec = [x / norm for x in vec]
        vectors.append(vec)

    return vocab, idf, vectors

# Example
corpus = [
    "oransight gpt improves slicing with rl",
    "rl for oran slicing with llm",
    "vector search aids rag for llm"
]
vocab, idf, X = build_tfidf(corpus)
# X is a list of TF-IDF vectors aligned with `vocab`
print("Vocabulary:", vocab)
print("IDF:", idf)
print("TF-IDF Vectors:")
for vec in X:
    print(vec)
# You can now use X for similarity search, clustering, etc.