##-- TF-IDF
from collections import Counter
import math 

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
        idf = {math.log((1+N)/(1+df[t])+1) for t in tokens}
    
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

