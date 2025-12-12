import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Clear GPU cache
gc.collect()
torch.cuda.empty_cache()

# Labels
LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt']

LABELS_TRAIN = [lbl for lbl in LABELS if lbl != 'scientific_debt']
LABEL2ID_TRAIN = {label: idx for idx, label in enumerate(LABELS_TRAIN)}
ID2LABEL_TRAIN = {v: k for k, v in LABEL2ID_TRAIN.items()}

model_hf_path = "bert-base-uncased"
batch_size = 128
lr = 0.0001
weight_decay = 0.1
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.texts = df['text'].fillna("").tolist()
        if is_train:
            self.labels = [LABEL2ID_TRAIN[label] for label in df['label']]
        else:
            self.labels = None
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

# Load data
df = pd.read_csv("/X/X/X/X-Project-1/models/master/dataset_analytics/master_4_artifact_dataset.csv")
df = df.dropna(subset=['text', 'label'])
df = df.groupby('label').filter(lambda x: len(x) >= 2).reset_index(drop=True)

# Split
df_scientific = df[df['label'] == 'scientific_debt'].reset_index(drop=True)
df_train = df[df['label'] != 'scientific_debt'].reset_index(drop=True)

print(f"Training on {len(df_train)} samples (5 classes)")
print(f"Evaluating on {len(df_scientific)} scientific_debt samples")

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_hf_path, num_labels=len(LABELS_TRAIN)
).to(device)

# DataLoaders
train_dataset = CommentDataset(df_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
    for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

# Evaluate on scientific_debt
sci_dataset = CommentDataset(df_scientific, tokenizer, is_train=False)
sci_loader = DataLoader(sci_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

sci_preds = []
model.eval()
with torch.no_grad():
    for input_ids, attention_mask in sci_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        sci_preds.extend(preds)

df_scientific['predicted_label'] = [ID2LABEL_TRAIN[p] for p in sci_preds]
df_scientific.to_csv("scientific_predictions.csv", index=False)
print("Saved scientific debt predictions to 'scientific_predictions.csv'")

# Show distribution
print("\nPrediction distribution on scientific_debt:")
print(df_scientific['predicted_label'].value_counts())
