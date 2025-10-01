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

def jaccard_similarity(doc1, doc2):
    """Compute Jaccard similarity between two documents (strings)."""
    set1 = set(doc1.lower().split())
    set2 = set(doc2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def cosine_similarity_sparse(a, b):
    """Compute cosine similarity between two sparse term-frequency vectors (dicts)."""
    num = sum(a.get(i,0)*b.get(i,0) for i in a.keys() | b.keys())
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else num/(na*nb)




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

# quicksort (1) : while
def sortArray(self, nums: List[int]) -> List[int]:
        
        def quicksort(l,r):
            if l>=r:
                return
            i = l
            pivot = nums[r]
            for j in range(l,r):
                if nums[j]<=pivot:
                    nums[i],nums[j]=nums[j],nums[i]
                    i +=1
            nums[i],nums[r] = nums[r],nums[i]
            quicksort(l,i-1)
            quicksort(i+1,r)
            

        quicksort(0,len(nums)-1)
        return nums

# quicksort (2) : for
def quicksort(l,r):
            if l>=r:
                return
            i = l
            pivot = nums[r]
            for j in range(l,r):
                if nums[j]<=pivot:
                    nums[i],nums[j]=nums[j],nums[i]
                    i +=1
            nums[i],nums[r] = nums[r],nums[i]
            quicksort(l,i-1)
            quicksort(i+1,r)
            

        quicksort(0,len(nums)-1)
###########################################
##########################################
##########################################

## LoRA simple code

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load base model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,              # LoRA rank
    lora_alpha=16,    # LoRA scaling
    target_modules=["c_attn"],  # Layer(s) to apply LoRA to (for GPT-2)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Example data
texts = ["Hello, how are you?", "LoRA is a parameter-efficient fine-tuning method."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Dummy training loop (for demonstration)
labels = inputs["input_ids"]
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
print("Loss:", loss.item())