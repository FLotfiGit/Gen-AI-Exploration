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




########## fun trick 
# rand10 from rand7: (efficient)

import random

def rand7():
    return random.randint(1, 7)

def rand10():
    while True:
        # Step 1: make 1..49
        a, b = rand7(), rand7()
        x = (a - 1) * 7 + b
        if x <= 40:
            return ((x - 1) % 10) + 1
        
        # Step 2: recycle overflow -> rand9()
        y = x - 40  # 1..9
        c = rand7()
        z = (y - 1) * 7 + c  # 1..63
        if z <= 60:
            return ((z - 1) % 10) + 1
        # otherwise loop again


######## simple backward:
import torch
import torch.nn as nn
import torch.functional as F

x = torch.randn(512,128)
y = torch.randn(512,1)
model = nn.Linear(128,1)
opt = torch.optim.SGD(model.parameters(),lr=0.1)

for step in range(2000):
    y_pred = model(x)
    loss = F.mse_loss(y_pred,y)

    opt.zero_grad()
    loss.backward()
    opt.step()


######## 

