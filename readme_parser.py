# readme_parser.py
import re
from dataclasses import dataclass, field
from typing import List
import markdown2

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')

@dataclass
class Node:
    level: int
    title: str
    content_lines: List[str] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)

    def add_line(self, line: str):
        self.content_lines.append(line)

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines).strip()

    def to_dict(self):
        return {
            "level": self.level, # 1 to 6
            "title": self.title.strip(),
            "html": markdown2.markdown(self.content, extras=["fenced-code-blocks","tables"]) if self.content else "",
            "children": [c.to_dict() for c in self.children],
        }

def parse_markdown_tree(md_text: str) -> List[Node]:
    lines = md_text.splitlines()
    root = Node(level=0, title="ROOT")
    stack = [root]
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2)
            node = Node(level=level, title=title)
            while stack and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].add_line(line)
    return root.children

def convert_markdown_to_sections(md_text: str):
    nodes = parse_markdown_tree(md_text)
    return [n.to_dict() for n in nodes]