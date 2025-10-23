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
            tokens = doc.lower().split()
            for t in tokens:
                df[t] +=1
        # Build vocab from all terms seen across docs
        self.vocab = {t:i for i, t in enumerate(sorted(df.keys()))}
        N = len(docs)
        # Smooth IDF
        self.idf = {t: math.log((1+N)/(1+df[t])) + 1.0 for t in self.vocab}
    
    def transform(self, docs):
        rows = []

        for doc in docs:
            tokens = doc.lower().split()
            tf = Counter(tokens)
            
            vec = {}
            for t, cnt in tf.items():
                if t in self.vocab:
                    i = self.vocab[t]
                    vec[i] = (cnt/len(tokens)) * self.idf[t]
            rows.append(vec)
        
        return rows
    
    def fit_transform(self,docs):
        self.fit(docs)
        return self.transform(docs)

    def get_feature_names(self):
        """Return feature terms ordered by their indices."""
        return [t for t, i in sorted(self.vocab.items(), key=lambda x: x[1])]

    def sparse_to_dense(self, vec, fill=0.0):
        """Convert a sparse vector (dict index->value) to a dense list."""
        dense = [fill] * len(self.vocab)
        for i, v in vec.items():
            dense[i] = v
        return dense

    def cosine_sparse(self, a: dict, b: dict) -> float:
        keys = a.keys() & b.keys()
        num = sum(a[i] * b[i] for i in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return 0.0 if na == 0 or nb == 0 else num / (na * nb)

    def topk_sparse(self, query_vec: dict, doc_vecs: list, k: int = 5):
        scores = [(i, self.cosine_sparse(query_vec, v)) for i, v in enumerate(doc_vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

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

def euclidean_distance_sparse(a, b):
    """Compute Euclidean distance between two sparse term-frequency vectors (dicts)."""
    keys = a.keys() | b.keys()
    return math.sqrt(sum((a.get(k,0)-b.get(k,0))**2 for k in keys))

def manhattan_distance_sparse(a, b):
    """Compute Manhattan distance between two sparse term-frequency vectors (dicts)."""
    keys = a.keys() | b.keys()
    return sum(abs(a.get(k,0)-b.get(k,0)) for k in keys)

def average_sparse_vector(vec):
    """Compute the average value of a sparse term-frequency vector (dict)."""
    if not vec:
        return 0.0
    return sum(vec.values()) / len(vec)




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
import torch.nn.functional as F

x = torch.randn(512,128)
y = torch.randn(512,1)
model = nn.Linear(128,1)
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2000):
    y_pred = model(x)
    loss = F.mse_loss(y_pred,y)

    opt.zero_grad()
    loss.backward()
    opt.step()

if __name__ == "__main__":
    # Tiny TFIDF demo
    docs = [
        "The quick brown fox",
        "Jumped over the lazy dog",
        "A quick brown dog"
    ]
    tfidf = TFIDF()
    vecs = tfidf.fit_transform(docs)
    q = "quick dog"
    q_vec = tfidf.transform([q])[0]
    top = tfidf.topk_sparse(q_vec, vecs, k=2)
    print("Top matches:", top)

######## 

from typing import List

# quicksort (1) : For
def sortArray(nums: List[int]) -> List[int]:

    def quicksort(l, r):
        if l >= r:
            return
        i = l
        pivot = nums[r]
        for j in range(l, r):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        quicksort(l, i - 1)
        quicksort(i + 1, r)

    quicksort(0, len(nums) - 1)
    return nums

# quickselect: Find the kth largest element in a list
def quickselect(nums, k):
    """
    Returns the kth largest element (1-based) in nums.
    """
    l, r = 0, len(nums) - 1
    k = len(nums) - k  # convert to kth smallest

    while l <= r:
        pivot = nums[r]
        i = l
        for j in range(l, r):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        if i == k:
            return nums[i]
        elif i < k:
            l = i + 1
        else:
            r = i - 1
###########################################
##########################################
##########################################

## LoRA simple code (optional)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
except Exception:
    AutoModelForCausalLM = AutoTokenizer = TrainingArguments = Trainer = None  # type: ignore

# Dynamically import PEFT to avoid linter errors when it's not installed
import importlib, importlib.util
_peft_spec = importlib.util.find_spec("peft")
if _peft_spec is not None:
    peft = importlib.import_module("peft")
    LoraConfig = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    prepare_model_for_kbit_training = getattr(peft, "prepare_model_for_kbit_training", None)
    _HAS_PEFT = all([LoraConfig, get_peft_model, prepare_model_for_kbit_training])
else:
    _HAS_PEFT = False

if _HAS_PEFT:
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
else:
    # Optional demo not run; PEFT/transformers not fully available.
    pass


################
# single head attention (toy snippet; Q, K, V are placeholders)
# Q, K, V expected shapes: (batch, seq_len, d)
# Uncomment and define Q,K,V before using
# d = Q.size(-1)
# score = (Q @ K.transpose(-2, -1)) / (d ** 0.5)
# weight = F.softmax(score, dim=-1)
# out = weight @ V