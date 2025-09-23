#Implement TF-IDF from scratch
import math
from collections import defaultdict, Counter

class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, docs):
        # build vocab & df
        df = Counter()
        for doc in docs:
            tokens = doc.lower().split()
            for t in set(tokens):
                df[t] += 1
        self.vocab = {t:i for i,t in enumerate(sorted(df))}
        N = len(docs)
        self.idf = {t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in self.vocab}  # smooth

    def transform(self, docs):
        rows = []
        for doc in docs:
            tokens = doc.lower().split()
            tf = Counter(tokens)
            vec = {}
            for t, cnt in tf.items():
                if t in self.vocab:
                    i = self.vocab[t]
                    vec[i] = (cnt / len(tokens)) * self.idf[t]
            rows.append(vec)  # sparse dict: index -> weight
        return rows

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

# Example
docs = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs"]
tfidf = TFIDF()
X = tfidf.fit_transform(docs)
print(tfidf.vocab)   # term -> index
print(X[0])          # sparse TF-IDF for doc0

#############################################
#############################################
#Cosine similarity search over TF-IDF
def cosine_sim_sparse(a, b):
    num = sum(a.get(i,0)*b.get(i,0) for i in a.keys() | b.keys())
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else num/(na*nb)

def topk_cosine(query_vec, doc_vecs, k=3):
    sims = [(i, cosine_sim_sparse(query_vec, v)) for i,v in enumerate(doc_vecs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

#############################################
#############################################
#BM25 scorer (classic IR baseline)
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [d.lower().split() for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs)/self.N
        self.k1, self.b = k1, b
        # df
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t]+=1

    def idf(self, t):
        # BM25+ style smooth idf
        return math.log(1 + (self.N - self.df.get(t,0) + 0.5)/(self.df.get(t,0)+0.5))

    def score(self, query, idx):
        q = query.lower().split()
        d = self.docs[idx]
        tf = Counter(d)
        score = 0.0
        for t in q:
            if t not in tf: 
                continue
            idf = self.idf(t)
            denom = tf[t] + self.k1*(1 - self.b + self.b*len(d)/self.avgdl)
            score += idf * (tf[t]*(self.k1+1))/denom
        return score

    def topk(self, query, k=3):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

# Example
docs = ["the cat sat on the mat", "the dog sat on the log", "cats are great pets"]
bm25 = BM25(docs)
print(bm25.topk("cat on mat", k=2))



#############################################
#############################################
#Sliding-window text chunker (RAG pre-processing)
def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks
