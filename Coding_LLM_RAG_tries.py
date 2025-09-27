import math
# Cosine similarity, norm 
def l2_norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [0.0]*len(v) if n==0 else [x/n for x in v]

def cosine(a,b):
    na, nb = l2_norm(a), l2_norm(b)
    return sum(x*y for x,y in zip(na,nb))

#---------------------------
# query q and list E, top-k similar?
def topk_cosine(q,E,k=5):
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
    out =[]
    i = 0

    while i<len(s):
        out.append([s[i:i+size]])
        if i+size >=len(s):
            break
        i += size - overlap
    return out
#---------------------------------------------



