import re
import json
from typing import List, Dict
from docx import Document
import pandas as pd
# import streamlit as st

# ======================================================
# Module 1: Input Loader
# ======================================================
def load_docx(path_or_file) -> Document:
    return Document(path_or_file)


# ======================================================
# Module 2: Low-level Parsers
# ======================================================
def para_text_from_element(p_elem) -> str:
    return ''.join(t.text for t in p_elem.xpath('.//w:t') if t.text).strip()


def extract_table_from_element(tbl_elem) -> List[List[str]]:
    table = []
    for row in tbl_elem.xpath('.//w:tr'):
        row_data = []
        for cell in row.xpath('.//w:tc'):
            ctext = ''.join(t.text for t in cell.xpath('.//w:t') if t.text).strip()
            row_data.append(ctext)
        if any(row_data):
            table.append(row_data)
    return table


# ======================================================
# Module 3A: Requirement Extractor
# ======================================================
def extract_requirements_from_doc(doc: Document) -> List[Dict]:
    FCT_MARKER = "FCT_"
    NUM_HEADING_RE = re.compile(r'^\s*((?:\d+\.)+\d+)\b')

    current_fct = None
    in_diag = False
    diag_prefix = None
    collected = []
    requirements = []

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
            "content": " ".join(content.split()),  # normalize spaces
            "diversity_expression": diversity
        })

    return requirements


def extract_content_and_diversity(table: List[List[str]]) -> (str, str):
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


# ======================================================
# Module 3B: Flow Extractor
# ======================================================
def extract_flows_from_doc(doc: Document) -> pd.DataFrame:
    start_heading = "FCT_"
    result = []
    current_heading = None

    # Step 1: Extract headings + tables
    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = element.xpath('.//w:t')
            text = ''.join([t.text for t in para if t.text])
            if start_heading in text:
                current_heading = text
        elif element.tag.endswith('tbl'):
            table = extract_table_from_element(element)
            result.append({"heading": current_heading, "table": table})

    # Step 2: Parse tables by type
    data = []
    for tab in result:
        if not tab['table']:
            continue

        first_cell = tab['table'][0][0]
        fct = tab['heading']

        if first_cell == "Flow Title":
            for row in tab['table'][1:]:
                data.append({"flowtitle": row[0], "heading": fct, "Status": "Flowtitles"})

        elif first_cell == "Flows":
            for row in tab['table'][1:]:
                data.append({"flowtitle": row[0], "heading": fct, "Status": "Internal"})

        elif first_cell == "PLM Parameter Name":
            for row in tab['table'][1:]:
                data.append({"flowtitle": row[2], "heading": fct, "Status": "PLM Parameters"})

        elif first_cell == "Parameter Name":
            for row in tab['table'][1:]:
                data.append({"flowtitle": row[0], "heading": fct, "Status": "CalibrationParameters"})

    df = pd.DataFrame(data)
    df = df[df['flowtitle'] != ""]
    df['FCT'] = df['heading'].apply(lambda x: x.split(":")[1] if ":" in x else x)
    return df


# ======================================================
# Module 4: Flow Validation with DCI Excel
# ======================================================
def get_from_wired(row, wired):
    if row['flowtitle'] in wired['Flow name'].values:
        match_rows = wired[wired['Flow name'] == row['flowtitle']]
        row_fct = str(row['FCT']).strip()
        matching_indices = match_rows['Allocated function'].astype(str).str.contains(row_fct, case=False, regex=False, na=False)
        if matching_indices.any():
            valid_match = match_rows[matching_indices]
            idx = valid_match.index[0]
            pc_val = str(wired.loc[idx, 'P/C'])
            inface = str(wired.loc[idx, 'Interface type(s)'])
            return f"{row['flowtitle']}_EX_{pc_val}#{inface}#*Wired"
        else:
            return f"{row['flowtitle']}_In*Wired"
    else:
        return f"{row['flowtitle']}_In*NA"


def get_flow_stat(row, netwk, wired):
    if row['flowtitle'] in netwk['Flow name'].values:
        match_rows = netwk[netwk['Flow name'] == row['flowtitle']]
        row_fct = str(row['FCT']).strip()
        matching_indices = match_rows['Allocated function'].astype(str).str.contains(row_fct, case=False, regex=False, na=False)
        if matching_indices.any():
            valid_match = match_rows[matching_indices]
            if not valid_match[['Frame name', 'Network']].isnull().all(axis=1).all():
                idx = valid_match.index[0]
                pc_val = str(netwk.loc[idx, 'P/C'])
                net_val = f"{netwk.loc[idx, 'Signal name']},{netwk.loc[idx, 'Frame name']}({netwk.loc[idx, 'Network']})"
                return f"{row['flowtitle']}_EX_{pc_val}_#{net_val}#*Network"
            else:
                return f"{row['flowtitle']}_In*Network"
        else:
            return get_from_wired(row, wired)

    elif row['flowtitle'] in wired['Flow name'].values:
        return get_from_wired(row, wired)

    else:
        return f"{row['flowtitle']}_In*NA"


def validate_flows(df: pd.DataFrame, dci_file) -> pd.DataFrame:
    netwk = pd.read_excel(dci_file, sheet_name='Network')
    wired = pd.read_excel(dci_file, sheet_name='Wired')

    df['var_status'] = df.apply(lambda x: get_flow_stat(x, netwk, wired), axis=1)
    df['External/internal'] = df['var_status'].apply(lambda x: "External" if "_EX_" in x else "Internal")
    df['P/C'] = df['var_status'].apply(lambda x: x.split("_EX_")[1][0] if "_EX_" in x else "In")
    df['sheet'] = df['var_status'].apply(lambda x: x.split("*")[1])
    df['Signal_and_Frame_Info'] = df['var_status'].apply(lambda x: x.split("#")[1] if "#" in x else "NA")
    return df


# ======================================================
# Streamlit UI
# ======================================================




# st.title("📑 Requirements & Flow Extractor")

# uploaded_doc = st.file_uploader("Upload Word Document (.docx)", type=["docx"])
# uploaded_dci = st.file_uploader("Upload DCI Excel (.xlsx)", type=["xlsx"])

# if uploaded_doc and uploaded_dci:
#     doc = load_docx(uploaded_doc)

#     st.subheader("🔹  Requirements Extraction")
#     requirements = extract_requirements_from_doc(doc)
#     st.success(f"Extracted {len(requirements)} requirements.")
#     st.json(requirements[:3])  # preview first 3

#     flows_df = extract_flows_from_doc(doc)
#     st.subheader("🔹 Flows Extraction from DCI")
#     validated_df = validate_flows(flows_df, uploaded_dci)
#     st.success(f"Mapped {len(validated_df)} flows.")

#     st.dataframe(validated_df.head())

#     # Download buttons
#     st.download_button(
#         "Download Requirements (JSON)",
#         json.dumps(requirements, indent=2, ensure_ascii=False),
#         file_name="requirements.json",
#         mime="application/json"
#     )

#     csv_bytes = validated_df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "Download Validated Flows (CSV)",
#         csv_bytes,
#         file_name="flows.csv",
#         mime="text/csv"
#     )

def flows_Extraction(uploaded_doc,uploaded_dci):
    doc_ = load_docx(uploaded_doc)
    flows_df = extract_flows_from_doc(doc_)
    validated_df = validate_flows(flows_df, uploaded_dci)
    validated_df.to_csv("flows.csv")
    return validated_df