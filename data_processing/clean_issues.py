import json
import re

INPUT_FILE = '/bsuhome/ericmelin/ORNL/ORNL-Project-1/hpc_scripts/all_issues.jsonl'
OUTPUT_FILE = 'all_issues_cleaned.jsonl'

def clean_text(text):
    if text is None:
        return ''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?!\s]", " ", text)  # Keep letters, ?, !, spaces
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text

with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for line in infile:
        artifact = json.loads(line)

        # Clean title and description
        artifact['source_sections']['title'] = clean_text(artifact['source_sections'].get('title'))
        artifact['source_sections']['description'] = clean_text(artifact['source_sections'].get('description'))

        # Clean each comment
        cleaned_comments = []
        for comment in artifact['source_sections']['comments']:
            cleaned_comment = {
                "user": comment['user'],
                "comment": clean_text(comment['comment']),
                "timestamp": comment['timestamp']
            }
            cleaned_comments.append(cleaned_comment)

        artifact['source_sections']['comments'] = cleaned_comments

        # Write cleaned artifact
        outfile.write(json.dumps(artifact) + '\n')

print(f"Preprocessing complete. Cleaned data saved to {OUTPUT_FILE}")
