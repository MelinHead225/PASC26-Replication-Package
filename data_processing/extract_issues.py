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

OUTPUT_FILE = 'all_issues.jsonl'

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
        print(f"Processing issues for repository: {repo_full_name}")
        repo = g.get_repo(repo_full_name)
        issues = repo.get_issues(state='all')  # open and closed issues

        for issue in tqdm(issues, desc=f'Extracting issues from {repo_full_name}', leave=False):
            # Skip pull requests
            if issue.pull_request is not None:
                continue

            comments = issue.get_comments()

            # Combine for reference extraction (raw, not cleaned)
            combined_text = (issue.title or '') + ' ' + (issue.body or '')
            for comment in comments:
                combined_text += ' ' + (comment.body or '')

            # Extract issue/PR references and URLs
            issue_pr_refs = list(set(re.findall(r"#\d+", combined_text)))
            urls = list(set(re.findall(r"https?://\S+", combined_text)))

            # Clean text fields
            cleaned_title = clean_github_text(issue.title or '')
            cleaned_body = clean_github_text(issue.body or '')

            artifact = {
                "project": repo_full_name.split('/')[1],
                "artifact_type": "Issue",
                "artifact_id": f"Issue_{issue.number}",
                "source_sections": {
                    "title": cleaned_title,
                    "description": cleaned_body,
                    "comments": []
                },
                "metadata": {
                    "author": issue.user.login if issue.user else None,
                    "created_at": issue.created_at.isoformat(),
                    "labels": [label.name for label in issue.labels],
                    "issue_pr_references": issue_pr_refs,
                    "linked_urls": urls
                },
                "source_link": issue.html_url
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
