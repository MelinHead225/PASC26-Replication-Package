import os
import re
import csv
import argparse

class CommentExtractor:
    def __init__(self):
        self.LANGUAGE_COMMENTS = {
            'python': {'ext': ['.py', '.pyx'], 'single': r'#(.*)', 'multi_start': r'^\s*(?:"""|\'\'\')(.*)', 'multi_end': r'(.*)(?:"""|\'\'\')\s*$'},
            'c_cpp': {'ext': ['.c', '.cpp', '.h', '.hpp'], 'single': r'//(.*)', 'multi_start': r'^\s*/\*(.*)', 'multi_end': r'(.*)\*/\s*$'},
            'fortran': {'ext': ['.f', '.for', '.f90'], 'single': r'!(.*)', 'multi_start': None, 'multi_end': None},
            'java': {'ext': ['.java'], 'single': r'//(.*)', 'multi_start': r'^\s*/\*(.*)', 'multi_end': r'(.*)\*/\s*$'},
            'shell': {'ext': ['.sh'], 'single': r'#(.*)', 'multi_start': None, 'multi_end': None},
            'cmake': {'ext': ['.cmake'], 'single': r'#(.*)', 'multi_start': None, 'multi_end': None},
            'matlab': {'ext': ['.m'], 'single': r'%(.*)', 'multi_start': None, 'multi_end': None},
            'rouge': {'ext': ['.rg'], 'single': r'(?:#|--)\s?(.*)', 'multi_start': None, 'multi_end': None},
        }
        self.encodings = ['utf-8', 'latin-1', 'utf-16']

    def is_license_comment(self, comment):
        license_keywords = ['license', 'copyright', 'distributed under']
        return any(keyword in comment.lower() for keyword in license_keywords)

    def clean_comment(self, comment):
        comment = re.sub(r'[^\w\s!?]', '', comment)
        comment = comment.lower()
        if self.is_license_comment(comment):
            return None
        return comment.strip()

    def get_language(self, file_extension):
        for lang, patterns in self.LANGUAGE_COMMENTS.items():
            if file_extension in patterns['ext']:
                return patterns
        return None

    def extract_comments_from_file(self, file_path):
        comments = []
        tried_encodings = []
        lines = None

        # Try UTF-8 explicitly first
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            tried_encodings.append('utf-8')

        # If UTF-8 failed, try other encodings
        if lines is None:
            for encoding in self.encodings:
                if encoding == 'utf-8':
                    continue  # Already tried
                try:
                    print(f"Trying to open {file_path} with encoding {encoding}")  # debug
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                    break
                except UnicodeDecodeError:
                    tried_encodings.append(encoding)

        if lines is None:
            raise RuntimeError(f"Unable to decode the file {file_path} with encodings {tried_encodings}")

        inside_multiline_comment = False
        multiline_comment = ""
        current_group_comment = []
        group_start_line = None

        file_extension = os.path.splitext(file_path)[1]
        lang_patterns = self.get_language(file_extension)

        if not lang_patterns:
            return []

        single_comment_pattern = lang_patterns['single']
        multi_start_pattern = lang_patterns['multi_start']
        multi_end_pattern = lang_patterns['multi_end']

        for line_number, line in enumerate(lines, 1):

            # Handle multiline comments first
            if inside_multiline_comment:
                multiline_comment += " " + line.strip()
                if multi_end_pattern and re.match(multi_end_pattern, line):
                    inside_multiline_comment = False
                    multiline_comment = multiline_comment.rstrip('*/').strip()
                    if multiline_comment:
                        comments.append((group_start_line, multiline_comment))
                    multiline_comment = ""
                continue

            # Detect start of multiline comment
            if multi_start_pattern:
                match_multi_start = re.match(multi_start_pattern, line)
                if match_multi_start:
                    inside_multiline_comment = True
                    multiline_comment = match_multi_start.group(1).strip()
                    group_start_line = line_number
                    # Check if multiline comment ends on the same line
                    if multi_end_pattern and re.match(multi_end_pattern, line):
                        inside_multiline_comment = False
                        multiline_comment = multiline_comment.rstrip('*/').strip()
                        if multiline_comment:
                            comments.append((group_start_line, multiline_comment))
                        multiline_comment = ""
                    continue

            # Handle single-line comments
            if single_comment_pattern:
                match_single = re.search(single_comment_pattern, line)
                if match_single:
                    comment = match_single.group(1).strip()

                    # Check if comment is inline (code before comment)
                    comment_start_pos = line.find(match_single.group(0))
                    code_before_comment = bool(line[:comment_start_pos].strip()) if comment_start_pos > 0 else False

                    if code_before_comment:
                        # Flush grouped full-line comments before adding this inline comment
                        if current_group_comment:
                            full_comment = ' '.join(current_group_comment).strip()
                            comments.append((group_start_line, full_comment))
                            current_group_comment = []
                            group_start_line = None

                        # Add inline comment separately
                        comments.append((line_number, comment))
                    else:
                        # This is a full-line comment
                        if not current_group_comment:
                            group_start_line = line_number
                        current_group_comment.append(comment)
                    continue

            # If line is not a comment line, flush any grouped full-line comments
            if current_group_comment:
                full_comment = ' '.join(current_group_comment).strip()
                comments.append((group_start_line, full_comment))
                current_group_comment = []
                group_start_line = None

        # After loop, flush any remaining grouped comments
        if current_group_comment:
            full_comment = ' '.join(current_group_comment).strip()
            comments.append((group_start_line, full_comment))

        return comments


    def traverse_multiple_repos_and_extract_comments(self, parent_dir):
        all_comments = []
        supported_extensions = [ext for patterns in self.LANGUAGE_COMMENTS.values() for ext in patterns['ext']]

        for repo_name in os.listdir(parent_dir):
            repo_path = os.path.join(parent_dir, repo_name)
            if os.path.isdir(repo_path):
                for dirpath, _, filenames in os.walk(repo_path):
                    for filename in filenames:
                        if any(filename.endswith(ext) for ext in supported_extensions):
                            file_path = os.path.join(dirpath, filename)
                            comments = self.extract_comments_from_file(file_path)
                            for line_number, comment in comments:
                                cleaned_comment = self.clean_comment(comment)
                                if cleaned_comment:
                                    all_comments.append([repo_name, file_path, line_number, cleaned_comment])
        return all_comments

    def save_comments_to_csv(self, comments, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Repository', 'File Path', 'Line Number', 'Comment'])
            csvwriter.writerows(comments)

def main():
    parser = argparse.ArgumentParser(description='Extract comments from multiple git repositories and save to a CSV file.')
    parser.add_argument('parent_dir', type=str, help='The parent directory containing git repositories')
    parser.add_argument('--output_file', type=str, default='all_comments.csv', help='The output CSV file (default: all_comments.csv)')

    args = parser.parse_args()

    extractor = CommentExtractor()
    comments = extractor.traverse_multiple_repos_and_extract_comments(args.parent_dir)
    extractor.save_comments_to_csv(comments, args.output_file)
    print(f"Comments with line numbers have been extracted and saved to {args.output_file}.")

if __name__ == "__main__":
    main()
