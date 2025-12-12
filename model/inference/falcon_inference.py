import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from tqdm import tqdm

# Labels and tasks
LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt']

ARTIFACT2ID = {
    "code_comments": 0,
    "commit_message": 1,
    "issue_section": 2,
    "pull_request_section": 2
}

# Model setup
repo_name = "MelinHead225/multitask-falcon-satd2"
tokenizer = AutoTokenizer.from_pretrained(repo_name)

class MultiTaskFalcon(nn.Module):
    def __init__(self, model_name, num_labels, num_tasks=4, pooling="last"):
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range(num_tasks)])
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, task_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_state = outputs.hidden_states[-1]

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

# Load model and heads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskFalcon(repo_name, num_labels=len(LABELS), num_tasks=len(ARTIFACT2ID)).to(device)

state_dict = torch.hub.load_state_dict_from_url(
    f"https://huggingface.co/{repo_name}/resolve/main/heads.pt",
    map_location=device
)
model.heads.load_state_dict(state_dict)
model.eval()

# Text cleaning functions
def remove_markdown_code(text):
    if not text:
        return ""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    return text

def remove_indented_code(text):
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not re.match(r"^\s{4,}", line)]
    return "\n".join(cleaned_lines)

def remove_stack_traces(text):
    if not text:
        return ""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if re.match(r'^\s*(at |File "|[A-Z_]+:|Exception)', line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def clean_text(text):
    text = remove_markdown_code(text)
    text = remove_indented_code(text)
    text = remove_stack_traces(text)
    return text.strip()

# Batch classification
def classify_batch(texts, artifact_source, max_len=101):
    task_id = ARTIFACT2ID[artifact_source]
    encoding = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    task_ids = torch.tensor([task_id] * len(texts), device=device)

    with torch.no_grad():
        logits = model(encoding["input_ids"], encoding["attention_mask"], task_ids)
    preds = torch.argmax(logits, dim=-1).tolist()
    return [(artifact_source, txt, LABELS[p]) for txt, p in zip(texts, preds)]

# JSONL inference + summaries
def analyze_jsonl(jsonl_path, out_csv="jsonl_predictions.csv", batch_size=16):
    all_items = []

    # Load and clean
    # with open(jsonl_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         try:
    #             obj = json.loads(line)
    #             text = clean_text(obj.get("text", ""))
    #             artifact_source = obj.get("artifact_type", "code_comments")
    #             if not text:
    #                 continue
    #             if artifact_source not in ARTIFACT2ID:
    #                 artifact_source = "code_comments"
    #             all_items.append((artifact_source, text))
    #         except json.JSONDecodeError:
    #             continue
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("artifact_type") != "pull_request_section":
                    continue   # skip everything except PRs
                text = clean_text(obj.get("text", ""))
                if not text:
                    continue
                all_items.append(("pull_request_section", text))
            except json.JSONDecodeError:
                continue


    print(f"Total items loaded: {len(all_items):,}")
    artifact_counts = pd.Series([a for a, _ in all_items]).value_counts()
    print("\n=== Artifact Type Counts ===")
    print(artifact_counts)

    # Inference
    results = []
    for i in tqdm(range(0, len(all_items), batch_size), desc="Running inference", unit="batch"):
        batch = all_items[i:i+batch_size]
        batch_sources, batch_texts = zip(*batch)
        # for src in set(batch_sources):
        #     src_texts = [txt for s, txt in batch if s == src]
        #     results.extend(classify_batch(src_texts, src))
        # Only one artifact type, pull_request_section
        batch_texts = [txt for _, txt in batch]
        results.extend(classify_batch(batch_texts, "pull_request_section"))

    # Create DataFrame
    df = pd.DataFrame(results, columns=["artifact_source", "text", "predicted_label"])

    # Save and Summarize
    df.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to: {out_csv}")

    # Stats
    print("\n=== Label Distribution (Overall) ===")
    label_counts = df["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="Count")
    label_counts["Percent"] = (label_counts["Count"] / len(df) * 100).round(2)
    print(label_counts.to_string(index=False))

    print("\n=== Artifact Source Distribution ===")
    print(df["artifact_source"].value_counts())

    grouped = (
        df.groupby(["artifact_source", "predicted_label"])
        .size()
        .reset_index(name="count")
    )
    grouped["total"] = grouped.groupby("artifact_source")["count"].transform("sum")
    grouped["percent"] = (grouped["count"] / grouped["total"] * 100).round(2)
    print("\n=== Label Distribution per Artifact Source ===")
    print(grouped.to_string(index=False))

    # Save summary files
    label_counts.to_csv("label_summary.csv", index=False)
    df["artifact_source"].value_counts().rename_axis("artifact_source").reset_index(name="count").to_csv("source_summary.csv", index=False)
    grouped.to_csv("label_by_artifact.csv", index=False)
    print("\nSaved: label_summary.csv, source_summary.csv, label_by_artifact.csv")

    return df

# Run
if __name__ == "__main__":
    jsonl_file = "/X/X/X/X-Project-1/models/master/cross-validation/falcon/all_artifacts.jsonl"
    df = analyze_jsonl(jsonl_file, "jsonl_predictions.csv", batch_size=64)
