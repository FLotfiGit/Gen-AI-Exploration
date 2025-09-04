import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

'''
QA over SQuAD dataset using finetuning of flan-T5
@ Flotfi 
Feb 2025
'''
# Load the SQuAD v2 dataset
dataset = load_dataset("squad_v2")

# Load the Flan-T5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Tokenization function
def preprocess_function(examples):
    inputs = [context + " question: " + question for question, context in zip(examples["question"], examples["context"])]
    targets = [answer['text'][0] if len(answer['text']) > 0 else "" for answer in examples["answers"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length', return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess datasets
train_dataset = dataset["train"].select(range(25000)).map(preprocess_function, batched=True)
validation_dataset = dataset["validation"].select(range(2000)).map(preprocess_function, batched=True)

# PyTorch dataset wrapper
class SQuAD2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.dataset[idx][key]) for key in ["input_ids", "attention_mask", "labels"]}
        return item

# Create PyTorch Dataloaders
batch_size = 64
train_loader = DataLoader(SQuAD2Dataset(train_dataset), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SQuAD2Dataset(validation_dataset), batch_size=batch_size, shuffle=False)

# Load pre-trained Flan-T5 model (PyTorch version)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training function
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
epochs = 3
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
model.save_pretrained("flan_t5_squad_v2_pytorch")
tokenizer.save_pretrained("flan_t5_squad_v2_pytorch")
