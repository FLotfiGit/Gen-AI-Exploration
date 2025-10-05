import math
# Cosine similarity, norm 
def l2_norm(v):
    """
    Compute the L2 norm (unit vector) of a list/vector.
    """
    n = math.sqrt(sum(x*x for x in v))
    return [0.0]*len(v) if n==0 else [x/n for x in v]

def cosine(a,b):
    """
    Compute cosine similarity between two vectors.
    """
    na, nb = l2_norm(a), l2_norm(b)
    return sum(x*y for x,y in zip(na,nb))

#---------------------------
# query q and list E, top-k similar?
def topk_cosine(q,E,k=5):
    """
    Return top-k most similar vectors to query q from list E using cosine similarity.
    """
    qn = l2_norm(q)
    sims = []
    for i,e in enumerate(E):
        en = l2_norm(e)
        sims.append((i,sum(x*y for x,y in zip(qn,en))))
    sims.sort(key=lambda x:x[1], reverse=True)
    return sims[:k]
#----------------------------
# MMR reranker (balance between relevant to q and diversity evidence)

def mmr(q, E, k=5,lam = 0.7):
    """
    Maximal Marginal Relevance (MMR) reranker: select k items balancing relevance to q and diversity.
    """
    chosen = []
    cand = set(range(len(E)))

    qn = l2_norm(q)
    En = [l2_norm(e) for e in E]

    def cos(i,j):
        return sum(x*y for x,y in zip(En[i],En[j]))
    def cosq(i):
        return sum(x*y for x,y in zip(qn,En[i]))
    
    while len(chosen)<k and cand:
        best, best_score = None, float('-inf')
        for i in cand:
            rel = cosq(i)
            div = 0.0 if not chosen else max(cos(i,j) for j in chosen)
            score = lam * rel - (1-lam)*div
            if score > best_score:
                best,best_score = i, score
        chosen.append(best)
        cand.remove(best)
    
    return chosen

#--------------------------------------------
# sliding window chunker (split into )

def chunk_text(s,size = 400, overlap=100):
    out = []
    i = 0
    while i < len(s):
        out.append(s[i:i+size])
        if i+size >= len(s):
            break
        i += size - overlap
    return out
#---------------------------------------------
# tokenizer trunk
def tokenize(text):
    """
    Simple whitespace tokenizer.
    """
    return text.strip().split()

def truncate_tokens(tokens, max_len):
    """
    Truncate token list to max_len tokens.
    """
    return tokens[:max_len]
#--------------------------------------------
# retrieval: sparse TF-IDF + cosine 
from collections import Counter

class TFIDF:
    def get_feature_names(self):
        """
        Return the list of feature names (vocabulary terms) in order of their indices.
        """
        return [t for t, i in sorted(self.vocab.items(), key=lambda x: x[1])]

    def sparse_to_dense(self, vec, fill=0.0):
        """
        Convert a sparse vector (dict) to a dense list using the vocab size.
        """
        dense = [fill] * len(self.vocab)
        for i, v in vec.items():
            dense[i] = v
        return dense

    def fit(self,docs):
        df = Counter()
        for d in docs:
            toks = d.lower().split()
            for t in toks: df[t] +=1
        self.vocab = {t:i for i,t in enumerate(sorted(df))}
        N = len(docs)
        self.idf = {t: math.log((1+N)/(1+df[t]))+1 for t in self.vocab}

    def transform(self,docs):
        rows = []
        for d in docs:
            toks = d.lower().split()
            tf = Counter(toks)
            vec = {}
            for t,c in tf.items():
                if t in self.vocab:
                    i = self.vocab[t]
                    vec[i] = (c/len(toks))*self.idf[t]
            rows.append(vec)
        return rows
    
    def cosine_sparse(self,a,b):
        keys = a.keys() & b.keys()
        num = sum(a[i]*b[i] for i in keys)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return 0 if na==0 or nb==0 else num/(na*nb)

    def topk_sparse(self,query_vec,doc_vecs, k=5):
        sc = [(i,self.cosine_sparse(query_vec,v)) for i, v in enumerate(doc_vecs)]
        sc.sort(key=lambda x:x[1],reverse=True)
        return sc[:k]

# -------------------
# Simple test/demo for TFIDF

if __name__ == "__main__":
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "A fox is quick and brown"
    ]
    tfidf = TFIDF()
    tfidf.fit(docs)
    vecs = tfidf.transform(docs)
    print("Feature names:", tfidf.get_feature_names())
    print("Sparse vector for doc 0:", vecs[0])
    print("Dense vector for doc 0:", tfidf.sparse_to_dense(vecs[0]))
    pass

    #-------------------------------------
    # End-to-End Mini RAG Retrieve-then-MMR :





    #-------------------------------------
    # Simple Passage Scorer with BM25 (RAG sparse fallback): 



    #--------------------------------------
    # toy beam search decoder :


