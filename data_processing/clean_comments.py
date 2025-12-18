import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?!\s]", " ", text)  # Keep letters, ?, !, spaces
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

input_file = '/bsuhome/ericmelin/ORNL/ORNL-Project-1/satd-data-augmentation/data-augmentation-pull-requests.csv'      
output_file = '/bsuhome/ericmelin/ORNL/ORNL-Project-1/models/master/dataset_analytics/pull-requests.csv'  

df = pd.read_csv(input_file, sep=';', encoding='ISO-8859-1')

if 'text' not in df.columns:
    raise ValueError("The CSV file must contain a 'text' column.")

df['text'] = df['text'].apply(clean_text)

df.to_csv(output_file, index=False)

print(f"Cleaned commenttext saved to '{output_file}'.")
