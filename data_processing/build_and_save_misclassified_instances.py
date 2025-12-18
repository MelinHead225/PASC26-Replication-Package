import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

gc.collect()
torch.cuda.empty_cache()

LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt'] 

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

model_hf_path = "bert-base-uncased"
batch_size = 256
lr = 0.0001
weight_decay = 0.1
num_epochs = 7       
n_splits = 5        

# File paths
labeled_path = "/master_4_artifact_dataset.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.texts = df['text'].fillna("").tolist()
        self.labels = [LABEL2ID[label] for label in df['label']] if is_train else None
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

        if self.labels is not None:
            return input_ids, attention_mask, self.labels[idx]
        else:
            return input_ids, attention_mask

def collate_fn(batch):
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        input_ids, attention_masks, labels = zip(*batch)
        return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)
    else:
        input_ids, attention_masks = zip(*batch)
        return torch.stack(input_ids), torch.stack(attention_masks)

df_labeled = pd.read_csv(labeled_path)

# Drop rows with missing text or label
df_labeled = df_labeled.dropna(subset=['text', 'label'])
# Keep only labels with at least 2 instances
df_labeled = df_labeled.groupby('label').filter(lambda x: len(x) >= 2).reset_index(drop=True)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store all misclassifications and aggregate predictions/labels 
all_misclassified = []
all_folds_preds = []
all_folds_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_labeled, df_labeled['label'])):
    print(f"\n===== Fold {fold+1}/{n_splits} =====")

    # Split data
    train_df = df_labeled.iloc[train_idx].reset_index(drop=True)
    val_df = df_labeled.iloc[val_idx].reset_index(drop=True)

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_hf_path, num_labels=len(LABELS)
    ).to(device)

    # Datasets & Loaders
    train_dataset = CommentDataset(train_df, tokenizer)
    val_dataset = CommentDataset(val_df, tokenizer)

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
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    # Validation 
    model.eval()
    fold_preds, fold_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            fold_preds.extend(preds)
            fold_labels.extend(labels.tolist())

    # Store for aggregate report
    all_folds_preds.extend(fold_preds)
    all_folds_labels.extend(fold_labels)

    # Classification report per fold
    print(f"Validation Report (Fold {fold+1}):")
    print(classification_report(fold_labels, fold_preds, target_names=LABELS, digits=4))

    # Store misclassified samples for this fold
    fold_results = val_df.copy()
    fold_results['true_label'] = [ID2LABEL[i] for i in fold_labels]
    fold_results['predicted_label'] = [ID2LABEL[i] for i in fold_preds]
    misclassified = fold_results[fold_results['true_label'] != fold_results['predicted_label']]
    all_misclassified.append(misclassified)

#  Combine all misclassified samples 
all_misclassified_df = pd.concat(all_misclassified, ignore_index=True)
all_misclassified_df.to_csv("all_misclassified_samples.csv", index=False)
print(f"\nSaved {len(all_misclassified_df)} misclassified samples to 'all_misclassified_samples.csv'")

#  Aggregate Classification Report across all folds 
print("\n===== Aggregate Classification Report Across All Folds =====")
print(classification_report(all_folds_labels, all_folds_preds, target_names=LABELS, digits=4))

#  Confusion Matrix for last fold 
plt.figure(figsize=(10, 8))
cm = confusion_matrix(fold_labels, fold_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Last Fold")
plt.tight_layout()
plt.savefig("confusion_matrix_last_fold.png")
plt.close()
print("Saved confusion matrix for last fold to 'confusion_matrix_last_fold.png'")
