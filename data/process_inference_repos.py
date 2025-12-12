from github import Github
import os, json, re
from tqdm import tqdm
from comment_extractor import CommentExtractor

# Config
GITHUB_TOKEN = "" # Insert Here!
g = Github(GITHUB_TOKEN)

# Cleaning
def clean_github_text(text):
    if not text:
        return ""
    # Remove fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    # Remove indented code blocks
    lines = [line for line in text.split("\n") if not re.match(r"^\s{4,}", line)]
    text = "\n".join(lines)
    # Remove stack traces
    cleaned = []
    for line in text.split("\n"):
        if re.match(r'^\s*(at |File "|[A-Z_]+:|Exception)', line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()

# Extraction Functions
def extract_code_comments(repo_path, projectname):
    extractor = CommentExtractor()
    cc_data = extractor.traverse_multiple_repos_and_extract_comments(repo_path)
    return [{
        "project": projectname,
        "artifact_type": "code_comments",
        "text": comment
    } for _, _, _, comment in cc_data]

def extract_commit_messages(repo_full_name, projectname):
    repo = g.get_repo(repo_full_name)
    commits = []
    for commit in tqdm(repo.get_commits(), desc=f"Commits {projectname}", leave=False):
        msg = clean_github_text(commit.commit.message or "")
        if msg:
            commits.append({
                "project": projectname,
                "artifact_type": "commit_message",
                "text": msg,
                "source_link": commit.html_url,
                "metadata": {
                    "author": commit.commit.author.name if commit.commit.author else None,
                    "created_at": commit.commit.author.date.isoformat() if commit.commit.author else None
                }
            })
    return commits

def extract_issues(repo_full_name, projectname):
    repo = g.get_repo(repo_full_name)
    artifacts = []
    for issue in tqdm(repo.get_issues(state="all"), desc=f"Issues {projectname}", leave=False):
        if issue.pull_request:
            continue

        # Title
        title_text = clean_github_text(issue.title or "")
        if title_text:
            artifacts.append({
                "project": projectname,
                "artifact_type": "issue_section",
                "text": title_text
            })

        # Body
        body_text = clean_github_text(issue.body or "")
        if body_text:
            artifacts.append({
                "project": projectname,
                "artifact_type": "issue_section",
                "text": body_text
            })

        # Comments
        for comment in issue.get_comments():
            comment_text = clean_github_text(comment.body or "")
            if comment_text:
                artifacts.append({
                    "project": projectname,
                    "artifact_type": "issue_section",
                    "text": comment_text
                })
    return artifacts

def extract_prs(repo_full_name, projectname):
    repo = g.get_repo(repo_full_name)
    artifacts = []
    for pr in tqdm(repo.get_pulls(state="all"), desc=f"PRs {projectname}", leave=False):
        # Title
        title_text = clean_github_text(pr.title or "")
        if title_text:
            artifacts.append({
                "project": projectname,
                "artifact_type": "pull_request_section",
                "text": title_text
            })

        # Body
        body_text = clean_github_text(pr.body or "")
        if body_text:
            artifacts.append({
                "project": projectname,
                "artifact_type": "pull_request_section",
                "text": body_text
            })

        # Comments
        for comment in pr.get_issue_comments():
            comment_text = clean_github_text(comment.body or "")
            if comment_text:
                artifacts.append({
                    "project": projectname,
                    "artifact_type": "pull_request_section",
                    "text": comment_text
                })
    return artifacts

def get_local_repos(repo_local_dir="/X/X/X/inference_projects"):
    repos = []
    for entry in os.listdir(repo_local_dir):
        full_path = os.path.join(repo_local_dir, entry)
        if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, ".git")):
            repos.append(entry)
    return repos

def process_repo(repo_full_name, repo_local_dir="/X/X/X/inference_projects"):
    projectname = repo_full_name.split("/")[1]
    repo_path = os.path.join(repo_local_dir, projectname)

    # Clone repo if not exists
    if not os.path.exists(repo_path):
        Repo.clone_from(f"https://github.com/{repo_full_name}.git", repo_path)

    all_data = []
    all_data += extract_code_comments(repo_path, projectname)
    all_data += extract_commit_messages(repo_full_name, projectname)
    all_data += extract_issues(repo_full_name, projectname)
    all_data += extract_prs(repo_full_name, projectname)
    return all_data

if __name__ == "__main__":
    local_repos = get_local_repos()
    REPO_MAP = {
        "Example": "Example/example",
    }
    all_results = []
    for short_name, repo_full_name in REPO_MAP.items():
        try:
            print(f"Processing {short_name} ({repo_full_name})")
            all_results.extend(process_repo(repo_full_name))
        except Exception as e:
            print(f"Failed on {short_name} ({repo_full_name}): {e}")


    with open("all_artifacts.jsonl", "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    print(f"Extracted {len(all_results)} artifacts total -> all_artifacts.jsonl")
