import re
import json
import argparse
from typing import List, Dict
from docx import Document


# -----------------------------
# Module 1: Input Loader
# -----------------------------
def load_docx(path_or_file) -> Document:
    """Load a Word document (.docx) into python-docx Document object."""
    return Document(path_or_file)


# -----------------------------
# Module 2: Parsers
# -----------------------------
def para_text_from_element(p_elem) -> str:
    """Extract concatenated text from a Word paragraph XML element."""
    return ''.join(t.text for t in p_elem.xpath('.//w:t') if t.text).strip()


def extract_table_from_element(tbl_elem) -> List[List[str]]:
    """Convert Word table XML element to 2D list of cell texts."""
    table = []
    for row in tbl_elem.xpath('.//w:tr'):
        row_data = []
        for cell in row.xpath('.//w:tc'):
            ctext = ''.join(t.text for t in cell.xpath('.//w:t') if t.text).strip()
            row_data.append(ctext)
        if any(row_data):  # ignore empty rows
            table.append(row_data)
    return table


# -----------------------------
# Text Cleaning Helpers
# -----------------------------
def clean_text_simple(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines into single space."""
    return re.sub(r'\s+', ' ', text).strip()


# -----------------------------
# Module 3: Requirement Extractor
# -----------------------------
def extract_requirements_from_doc(doc: Document) -> List[Dict]:
    """Extract requirements from the uploaded Word document."""
    FCT_MARKER = "FCT_"
    NUM_HEADING_RE = re.compile(r'^\s*((?:\d+\.)+\d+)\b')

    current_fct = None
    in_diag = False
    diag_prefix = None
    collected = []
    requirements = []

    # ---- Step 1: Walk document ----
    for element in doc.element.body:
        if element.tag.endswith('p'):
            text = para_text_from_element(element)
            if not text:
                continue
            if FCT_MARKER in text:
                current_fct = text
                in_diag = False
                diag_prefix = None
            m = NUM_HEADING_RE.match(text)
            if "Diagnostic Requirements" in text:
                in_diag = True
                diag_prefix = m.group(1) if m else None
                continue
            if in_diag and m:
                heading_num = m.group(1)
                if diag_prefix and not heading_num.startswith(diag_prefix):
                    in_diag = False
                    diag_prefix = None
                elif not diag_prefix:
                    in_diag = False
        elif element.tag.endswith('tbl'):
            if not current_fct or in_diag:
                continue
            table = extract_table_from_element(element)
            if not table:
                continue
            collected.append({"heading": current_fct, "table": table})

    # ---- Step 2: Refine into requirement objects ----
    for block in collected:
        table = block["table"]
        first_cell = table[0][0] if table and table[0] else ""
        if "REQ-" not in first_cell:
            continue
        parts = first_cell.split()
        req_id = parts[0]
        if len(parts) > 1:
            req_id += "  " + parts[1]
        fct_name = None
        if len(parts) >= 3:
            for token in parts:
                if token.startswith("Fct_"):
                    fct_name = token
        heading_clean = block["heading"].replace("Effectivity of FA :", "").strip()
        content, diversity = extract_content_and_diversity(table)
        requirements.append({
            "req_id": req_id,
            "fct_name": fct_name or heading_clean,
            "content": clean_text_simple(content),
            "diversity_expression": clean_text_simple(diversity)
        })

    return requirements


def extract_content_and_diversity(table: List[List[str]]) -> (str, str):  # type: ignore
    """Extract requirement content and diversity expression from table."""
    content = ""
    diversity = ""
    for i, row in enumerate(table):
        row_join = " ".join(row)
        if "Content of the Requirement" in row_join:
            content_lines = []
            for j in range(i + 1, len(table)):
                next_row = table[j]
                if any("Expression" in cell or "Description" in cell for cell in next_row):
                    break
                line = " ".join(next_row).strip()
                if line:
                    line = line.replace("·", "- ")
                    content_lines.append(line)
            content = "\n".join(content_lines).strip()
        if "Diversity Expression" in row_join and i + 1 < len(table):
            diversity = " ".join(table[i + 1]).strip()
    return content, diversity


# -----------------------------
# CLI mode only
# -----------------------------
def run_cli(doc_path: str, out_path: str):
    doc = load_docx(doc_path)
    requirements = extract_requirements_from_doc(doc)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(requirements)} requirements → {out_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Requirement Preprocessor")
#     parser.add_argument("--doc", help="Path to input Word document (.docx)")
#     parser.add_argument("--out", default="raw_non_diagnostic_requirements.json", help="Output JSON file")
#     args = parser.parse_args()
#     run_cli(args.doc, args.out)
