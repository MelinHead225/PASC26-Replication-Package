import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
import gc

from transformers import AutoTokenizer

print(f"HF cache: {getattr(__import__('os'), 'environ')['HF_HOME']}")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-3B-Instruct")


# Clear GPU cache
gc.collect()
torch.cuda.empty_cache()

# Configuration
LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt'] 

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# model_hf_path = "bert-base-uncased"
# model_hf_path = "roberta-base"
# model_hf_path = "microsoft/deberta-v3-base"
# model_hf_path = "allenai/scibert_scivocab_uncased"
# model_hf_path = "microsoft/codebert-base"
model_hf_path = "tiiuae/Falcon3-3B-Instruct"
# model_hf_path = "meta-llama/Llama-3.2-1B"
# model_hf_path = "mistralai/Mistral-7B-v0.3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class CommentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].fillna("").tolist()
        self.labels = [LABEL2ID[label] for label in df['label']]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, self.labels[idx]

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)

# Load labeled data
labeled_path = "/X/X/X/X-Project-1/models/master/dataset_analytics/master_4_artifact_dataset.csv"
df_labeled = pd.read_csv(labeled_path)
df_labeled = df_labeled.dropna(subset=['text', 'label'])
df_labeled = df_labeled.groupby('label').filter(lambda x: len(x) >= 2).reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(model_hf_path)

# If no pad token, fall back to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_hf_path, num_labels=len(LABELS)
).to(device)

# Resize token embeddings to account for new pad token
model.resize_token_embeddings(len(tokenizer))

# Train/Val split (80/20)
train_size = int(0.8 * len(df_labeled))
val_size = len(df_labeled) - train_size
train_df, val_df = random_split(df_labeled, [train_size, val_size])

train_dataset = CommentDataset(pd.DataFrame(train_df.dataset.iloc[train_df.indices]), tokenizer)
val_dataset = CommentDataset(pd.DataFrame(val_df.dataset.iloc[val_df.indices]), tokenizer)

# Hyperparameter grid
param_grid = {
    "batch_size": [256],
    "lr": [1e-5, 5e-5, 1e-4],
    "weight_decay": [0.0, 0.01, 0.1],
    "num_epochs": [5]
}

best_score = 0
best_params = {}

# Grid search loop
for batch_size in param_grid['batch_size']:
    for lr in param_grid['lr']:
        for weight_decay in param_grid['weight_decay']:
            for num_epochs in param_grid['num_epochs']:
                print(f"\n=== Trying: batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}, epochs={num_epochs} ===")

                # Model
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_hf_path, num_labels=len(LABELS)
                ).to(device)

                # Set pad_token_id inside the model’s config
                model.config.pad_token_id = tokenizer.pad_token_id

                # Resize embeddings for pad token
                model.resize_token_embeddings(len(tokenizer))
              
                # DataLoaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                # Optimizer & Scheduler
                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = get_scheduler(
                    "linear", optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(train_loader) * num_epochs
                )
                criterion = torch.nn.CrossEntropyLoss()

                # Training
                model.train()
                for epoch in range(num_epochs):
                    total_loss = 0
                    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        total_loss += loss.item()
                    # print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

                # Validation
                from sklearn.metrics import classification_report

                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for input_ids, attention_mask, labels in val_loader:
                        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                        outputs = model(input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                        val_preds.extend(preds)
                        val_labels.extend(labels.tolist())

                # Classification Report
                report = classification_report(val_labels, val_preds, target_names=LABELS, digits=4)
                print("Validation Classification Report:")
                print(report)

                # Compute overall accuracy
                acc = sum([p==t for p, t in zip(val_preds, val_labels)]) / len(val_labels)
                print(f"Validation Accuracy: {acc:.4f}")


                # Track best
                if acc > best_score:
                    best_score = acc
                    best_params = {
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "num_epochs": num_epochs
                    }

print("\n=== Best Hyperparameters ===")
print(best_params)
print(f"Best Validation Accuracy: {best_score:.4f}")
