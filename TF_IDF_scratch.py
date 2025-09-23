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
# Each vector corresponds to a document in the corpus

def jaccard_similarity(doc1, doc2):
    """Compute Jaccard similarity between two documents (strings)."""
    set1 = set(tokenize(doc1))
    set2 = set(tokenize(doc2))
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two TF-IDF vectors."""
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two TF-IDF vectors."""
    return math.sqrt(sum((a-b)**2 for a, b in zip(vec1, vec2)))

def cluster_documents_kmeans(vectors, n_clusters=2):
    """Cluster TF-IDF vectors using KMeans."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("Please install scikit-learn.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    return labels

def manhattan_distance(vec1, vec2):
    """Compute Manhattan distance between two TF-IDF vectors."""
    return sum(abs(a-b) for a, b in zip(vec1, vec2))

def average_vector(vectors):
    """Compute the average TF-IDF vector for a list of vectors."""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    avg = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            avg[i] += val
    return [x / n for x in avg]
    
def top_idf_terms(idf, top_n=5):
    """Return the top N terms with highest IDF values."""
    return sorted(idf.items(), key=lambda x: x[1], reverse=True)[:top_n]
