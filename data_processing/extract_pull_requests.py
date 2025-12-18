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

OUTPUT_FILE = 'all_pr.jsonl'

def remove_markdown_code(text):
    if not text:
        return ""
    # Remove fenced code blocks (```...```)
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code (`...`)
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

def clean_github_text(text):
    text = remove_markdown_code(text)
    text = remove_indented_code(text)
    text = remove_stack_traces(text)
    return text.strip()

g = Github(GITHUB_TOKEN)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for repo_full_name in REPO_NAMES:
        print(f"Processing pr for repository: {repo_full_name}")
        repo = g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state='all')  # open and closed PRs

        for pr in tqdm(pulls, desc=f'Extracting PRs from {repo_full_name}', leave=False):
            comments = pr.get_issue_comments()

            combined_text = (pr.title or '') + ' ' + (pr.body or '')
            for comment in comments:
                combined_text += ' ' + (comment.body or '')

            issue_pr_refs = list(set(re.findall(r"#\d+", combined_text)))
            urls = list(set(re.findall(r"https?://\S+", combined_text)))

            cleaned_title = clean_github_text(pr.title or '')
            cleaned_body = clean_github_text(pr.body or '')

            artifact = {
                "project": repo_full_name.split('/')[1],
                "artifact_type": "PullRequest",
                "artifact_id": f"PR_{pr.number}",
                "source_sections": {
                    "title": cleaned_title,
                    "description": cleaned_body,
                    "comments": []
                },
                "metadata": {
                    "author": pr.user.login if pr.user else None,
                    "created_at": pr.created_at.isoformat(),
                    "labels": [label.name for label in pr.labels],
                    "issue_pr_references": issue_pr_refs,
                    "linked_urls": urls
                },
                "source_link": pr.html_url
            }

            for comment in comments:
                cleaned_comment = clean_github_text(comment.body or '')
                artifact["source_sections"]["comments"].append({
                    "user": comment.user.login if comment.user else None,
                    "comment": cleaned_comment,
                    "timestamp": comment.created_at.isoformat()
                })

            f.write(json.dumps(artifact) + '\n')


print(f"Finished extracting issues for {len(REPO_NAMES)} repositories to {OUTPUT_FILE}")
