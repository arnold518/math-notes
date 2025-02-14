import re
import os
from pathlib import Path
from content import *
from markdown_parser import MarkdownParser

title = ""
filepath = "../docs/MFDNN/"
format_markdown_filepath = filepath + 'format_markdown/'
format_latex_filepath = filepath + 'format_latex/'

class Chapter:
    def __init__(self, title="", number=0):
        self.title = title
        self.number = number
        self.sections = []

class Section:
    def __init__(self, title="", number=0, filepath=""):
        self.title = title
        self.number = number  # 0 for unnumbered sections
        self.filepath = filepath
        self.filename = Path(self.filepath).stem
        self.content = ContentList()
        self.str_markdown = ""
        self.str_latex = ""
    def read_markdown(self):
        if not self.filepath.endswith('.md'): return False
        parser = MarkdownParser(filepath + self.filepath)
        self.content = parser.parse()
        return True
    def format_markdown(self):
        with open(format_markdown_filepath + self.filename + '.md', 'w') as file:
            self.str_markdown = self.content.str_markdown()
            file.write(self.str_markdown)
    def format_latex(self):
        with open(format_latex_filepath + self.filename + '.tex', 'w') as file:
            self.str_latex = self.content.str_latex()
            file.write(self.str_latex)

def parse_toc(file_path):
    global title
    chapters = []
    current_chapter = None

    with open(file_path, "r", encoding="utf-8") as f:
        flag = False
        for line in f:
            line = line.strip()

            if line.startswith('# '):
                title = line[2:].strip()
            if line.startswith('## '):
                if 'Table of Contents' in line:
                    flag = True
                else:
                    flag = False

            if not flag : continue

            # Match chapter titles (e.g., "### Ch 1. Optimization and Stochastic Gradient Descent")
            ch_match = re.match(r"### Ch ([A-Za-z0-9]+)\. (.+)", line)
            if ch_match:
                chapter_number = ch_match.group(1)
                chapter_title = ch_match.group(2)

                # Convert chapter_number to an integer if possible, otherwise leave as 0 for unnumbered chapters
                try:
                    chapter_number = int(chapter_number)
                except ValueError:
                    chapter_number = 0  # Handle appendices or unnumbered chapters

                current_chapter = Chapter(title=chapter_title, number=chapter_number)
                chapters.append(current_chapter)
                continue

            # Match numbered sections (e.g., "- [1. Optimization Problem](1.md)")
            sec_match = re.match(r"- \[(\d+)\. (.+?)\]\((.+?)\)", line)
            if sec_match and current_chapter:
                section_number = int(sec_match.group(1))
                section_title = sec_match.group(2)
                section_filepath = sec_match.group(3)

                section = Section(title=section_title, number=section_number, filepath=section_filepath)
                current_chapter.sections.append(section)
                continue

            # Match unnumbered sections (e.g., "- [Introduction](intro.md)")
            unnumbered_sec_match = re.match(r"- \[(.+?)\]\((.+?)\)", line)
            if unnumbered_sec_match and current_chapter:
                section_title = unnumbered_sec_match.group(1)
                section_filepath = unnumbered_sec_match.group(2)

                section = Section(title=section_title, number=0, filepath=section_filepath)
                current_chapter.sections.append(section)

    return chapters

def merge_latex(chapters):
    merged = ""
    with open("latex_prefix.tex", "r", encoding="utf-8") as file:
        merged = file.read()
    merged += "\\begin{document}\n\n"
    merged += f"\\title{{{title}}}\n"
    merged += "\\author{}\n\\date{}\n\\maketitle\n\\tableofcontents\n\n"

    for chapter in chapters:
        if chapter.number == 0 :
            merged += f"\\part*{{{chapter.title}}}\n\n"
        else :
            merged += f"\\part{{{chapter.title}}}\n\n"
        for section in chapter.sections:
            if section.str_latex == "" :
                if section.number == 0 :
                    merged += f"\\chapter*{{{section.title}}}\n\n"
                else :
                    merged += f"\\chapter{{{section.title}}}\n\n"
                merged += f"\\href{{{section.filepath}}}{{{section.title}}}\n\n"
            else :
                merged += section.str_latex + "\n"
                pass
            
    merged += "\\end{document}\n"

    with open(format_latex_filepath + 'merged_latex.tex', 'w') as file:
        file.write(merged)

    return merged


chapters = parse_toc(filepath + "index.md")    

if not os.path.exists(format_markdown_filepath):
    os.makedirs(format_markdown_filepath)

if not os.path.exists(format_latex_filepath):
    os.makedirs(format_latex_filepath)

for chapter in chapters:
    for section in chapter.sections:
        if section.read_markdown() :
            section.format_markdown()
            section.format_latex()

for chapter in chapters:
    print(f"Chapter {chapter.number}: {chapter.title}")
    for section in chapter.sections:
        section_label = f"Section {section.number}"
        print(f"  {section_label}: {section.title} ({section.filepath})")

merge_latex(chapters)