# Find top 100 users in a repo for prs/issues. Then manually check for bots in the list to remove.
import json
from collections import defaultdict, Counter

INPUT_FILE = '/bsuhome/ericmelin/ORNL-Project-1/hpc_scripts/all_issues_cleaned.jsonl'
# INPUT_FILE = '/bsuhome/ericmelin/ORNL-Project-1/hpc_scripts/all_prs_cleaned.jsonl'

OUTPUT_FILE = 'top_100_users_per_repo.json'

# Data structure: {repo_name: Counter({user: comment_count})}
repo_user_comment_counts = defaultdict(Counter)

# Read the JSONL file and count user comments
with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
    for line in infile:
        artifact = json.loads(line)
        repo_name = artifact['project']
        comments = artifact['source_sections']['comments']

        for comment in comments:
            user = comment['user']
            if user:  # Skip empty user fields
                repo_user_comment_counts[repo_name][user] += 1

# Extract top 100 users per repository
top_users_per_repo = {}
for repo, user_counter in repo_user_comment_counts.items():
    top_users_per_repo[repo] = user_counter.most_common(100)

# Save to JSON for easy review
with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    json.dump(top_users_per_repo, outfile, indent=4)

print(f"Top 100 users per repository saved to {OUTPUT_FILE}")
