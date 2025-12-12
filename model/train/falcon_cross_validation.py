import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc

# Clear GPU cache
gc.collect()
torch.cuda.empty_cache()

# Configuration
LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt']
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ARTIFACTS = ['code_comments', 'commit_message', 'issue_section', 'pull_request_section']
ARTIFACT2ID = {a: idx for idx, a in enumerate(ARTIFACTS)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
labeled_path = "/X/X/X/X-Project-1/models/master/dataset_analytics/master_4_artifact_dataset.csv"
df = pd.read_csv(labeled_path)
df = df.dropna(subset=['text', 'label', 'artifact_source'])
df = df.groupby('label').filter(lambda x: len(x) >= 2).reset_index(drop=True)

# Tokenizer
model_hf_path = "tiiuae/Falcon3-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset class
class MultiTaskCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=101):
        self.texts = df['text'].fillna("").tolist()
        self.labels = [LABEL2ID[label] for label in df['label']]
        self.artifact_ids = [ARTIFACT2ID[src] for src in df['artifact_source']]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, self.labels[idx], self.artifact_ids[idx]

def collate_fn(batch):
    input_ids, attention_masks, labels, artifact_ids = zip(*batch)
    return (torch.stack(input_ids),
            torch.stack(attention_masks),
            torch.tensor(labels),
            torch.tensor(artifact_ids))

# Multi-task Falcon model
class MultiTaskFalcon(nn.Module):
    def __init__(self, model_name, num_labels, num_tasks=4, pooling="last"):
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range(num_tasks)])
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, task_ids):
        outputs = self.encoder.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        if self.pooling == "last":
            pooled_output = hidden_state[:, -1, :]
        elif self.pooling == "mean":
            pooled_output = (hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            raise ValueError("Unsupported pooling type")

        logits = torch.zeros(input_ids.size(0), self.heads[0].out_features, device=input_ids.device)
        for task_idx in range(len(self.heads)):
            mask = (task_ids == task_idx)
            if mask.any():
                logits[mask] = self.heads[task_idx](pooled_output[mask])
        return logits

# Hyperparameters
batch_size = 128
lr = 1e-5
weight_decay = 0.01
num_epochs = 3
max_len = 101
n_splits = 3

# Stratified K-Fold
X = df['text'].values
y = df['label'].values
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_reports = []
all_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n=== Fold {fold} ===")
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_dataset = MultiTaskCommentDataset(train_df, tokenizer, max_len=max_len)
    val_dataset = MultiTaskCommentDataset(val_df, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskFalcon(model_hf_path, num_labels=len(LABELS), num_tasks=len(ARTIFACTS)).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader)*num_epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels, task_ids in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}"):
            input_ids, attention_mask, labels, task_ids = (
                input_ids.to(device), attention_mask.to(device),
                labels.to(device), task_ids.to(device)
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, task_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, task_ids in val_loader:
            input_ids, attention_mask, labels, task_ids = (
                input_ids.to(device), attention_mask.to(device),
                labels.to(device), task_ids.to(device)
            )
            outputs = model(input_ids, attention_mask, task_ids)
            preds = torch.argmax(outputs, dim=-1).cpu().tolist()
            val_preds.extend(preds)
            val_labels.extend(labels.tolist())

    report = classification_report(val_labels, val_preds, target_names=LABELS, digits=4, output_dict=True)
    all_reports.append(report)
    acc = np.mean([p==t for p,t in zip(val_preds,val_labels)])
    all_accuracies.append(acc)
    print(f"Fold {fold} Accuracy: {acc:.4f}")

# Aggregate metrics
print("\n=== Cross-Validation Results ===")
print(f"Average Accuracy: {np.mean(all_accuracies):.4f}")
macro_f1s = [rep['macro avg']['f1-score'] for rep in all_reports]
print(f"Average Macro F1: {np.mean(macro_f1s):.4f}")

# Aggregate per-class metrics
per_class = {label: {'precision': [], 'recall': [], 'f1-score': []} for label in LABELS}
for rep in all_reports:
    for label in LABELS:
        for metric in ['precision','recall','f1-score']:
            per_class[label][metric].append(rep[label][metric])

print("\n=== Aggregated Per-Class Metrics ===")
for label in LABELS:
    print(f"{label}:")
    print(f"  Precision: {np.mean(per_class[label]['precision']):.4f}")
    print(f"  Recall:    {np.mean(per_class[label]['recall']):.4f}")
    print(f"  F1-score:  {np.mean(per_class[label]['f1-score']):.4f}")
