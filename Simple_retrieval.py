from Coding_exercise import TFIDF
import math
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
    return [owners[i], all_chunks[i], score for i,score in ranked]



