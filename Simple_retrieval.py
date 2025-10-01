# from coding_camp import TFIDF
import math
from collections import Counter
#: Utils 

# chunk each string (sliding window)
def chunk_text(s, chunk_size=200, overlap=50):
    chunk = []
    i = 0

    while i<len(s):
        chunk.append(s[i:i+chunk_size])
        if i+chunk_size > len(s):
            break
        i +=chunk_size-overlap
    
    return chunk

# cosine similarity over TF-IDF
def cosine_sim_sparse(A,B):
  num = sum(A.get(i,0)*B.get(i,0) for i in A.keys() | B.keys())
  na = math.sqrt(sum(v*v for v in A.values()))
  nb = math.sqrt(sum(v*v for v in B.values()))
  return 0 if na==0 or nb==0 else num/(na*nb)


def top_cosine(query_vec, doc_vec, k):
    sims = [(i, cosine_sim_sparse(query_vec, v)) for i,v in enumerate(doc_vec)]
    sims.sort(key=lambda x:x[1], reverse=True)
    return sims[:k]

################# TFIDF class
##-- TF-IDF

class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.idf ={}
    
    def fit(self,docs):
        df = Counter()

        for doc in docs:
            tokens = doc.lower.split()
            for t in tokens:
                df[t] +=1
        self.vocab = {t:i for t,i in enumerate(sorted(tokens))}
        N = len(docs)
        self.idf = {math.log((1+N)/(1+df[t])+1) for t in tokens}
    
    def transform(self, docs):
        rows ={}

        for doc in docs:
            tokens = doc.lower.split()
            tf = Counter(tokens)
            
            vec ={}
            for t, cnt in tf.items():
                i = self.vocab[t]
                vec[i] = (cnt/len(tokens)) * self.idf[t]
                rows.append(vec)
        
        return rows
    
    def fit_transform(self,docs):
        self.fit(docs)
        return self.transform(docs)


################ retrieval docs #################

def simple_retrieve(docs, query, chunk_size=200, overlap=50, k=3):
    all_chunks =[]
    owners = []
    # (1): chunking
    for i,d in enumerate(docs):
        chunks = chunk_text(d, chunk_size, overlap)
        all_chunks.extend(chunks)
        owners.extend([i] * len(chunks))
    
    # (2): TF-IDF
    vec = TFIDF()
    X = vec.fit_transform(all_chunks)
    qv = vec.transform([query])[0]

    # (3): ranking
    ranked = top_cosine(X,qv,k)
    return [(owners[i], all_chunks[i], score) for i,score in ranked]



################## compare sparse vs dense retrieval ###########
# --------------------------
# TF-IDF vs Embedding Retrieval
# --------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Example corpus
docs = [
    "The doctor prescribed medicine for the patient.",
    "The physician gave treatment in the hospital.",
    "The engineer designed a new bridge.",
    "A chef cooked pasta in the kitchen."
]

query = "medical doctor"

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs + [query])  # fit on all docs + query
query_vec = tfidf_matrix[-1]
doc_vecs = tfidf_matrix[:-1]

tfidf_scores = cosine_similarity(query_vec, doc_vecs).flatten()

print("TF-IDF Scores:")
for i, score in enumerate(tfidf_scores):
    print(f"Doc {i}: {score:.4f}")

# ---------- Embeddings ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the [CLS] token embedding as sentence representation
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

doc_embeddings = torch.stack([get_bert_embedding(doc) for doc in docs])
query_embedding = get_bert_embedding(query)

embedding_scores = (doc_embeddings @ query_embedding).numpy()
# (dot product since embeddings are normalized in this model)

print("\nEmbedding Scores:")
for i, score in enumerate(embedding_scores):
    print(f"Doc {i}: {score:.4f}")
