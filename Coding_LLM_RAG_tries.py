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
