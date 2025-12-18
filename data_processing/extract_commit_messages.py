from github import Github
import json
from tqdm import tqdm
import re

GITHUB_TOKEN = ''

REPO_NAMES = [
    'ornladios/ADIOS2',
    'visit-dav/visit',
    'dyninst/dyninst',
    'UO-OACISS/tau2',
    'hypre-space/hypre',
    'trilinos/Trilinos',
    'kokkos/kokkos',
    'StanfordLegion/legion',
    'spack/spack'
]

OUTPUT_FILE = 'all_commit_messages.jsonl'

# Github Connection
g = Github(GITHUB_TOKEN)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for repo_full_name in REPO_NAMES:
        print(f"Processing repository: {repo_full_name}")
        repo = g.get_repo(repo_full_name)
        commits = repo.get_commits()
        
        for commit in tqdm(commits, desc=f'Extracting commits for {repo_full_name}', leave=False):
            commit_data = commit.commit

            raw_message = commit_data.message or ""

            # Extract issue or PR references like #1234
            issue_refs = re.findall(r"#\d+", raw_message)

            # Extract URLs
            urls = re.findall(r"https?://\S+", raw_message)

            # Lowercase and clean message, but keep only text characters, spaces, '!', '?'
            clean_message = raw_message.lower()
            clean_message = re.sub(r"[^a-zA-Z\s!?]", "", clean_message)

            artifact = {
                "project": repo_full_name.split('/')[1],  
                "artifact_type": "Commit",
                "artifact_id": f"Commit_{commit.sha[:7]}",
                "source_sections": {
                    "message": clean_message
                },
                "metadata": {
                    "author": commit_data.author.name if commit_data.author else None,
                    "created_at": commit_data.author.date.isoformat() if commit_data.author else None,
                    "issue_references": issue_refs,
                    "linked_urls": urls
                },
                "source_link": commit.html_url
            }

            f.write(json.dumps(artifact) + '\n')

print(f"Finished extracting commits for {len(REPO_NAMES)} repositories to {OUTPUT_FILE}")
