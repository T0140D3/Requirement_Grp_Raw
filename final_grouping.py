import json
import os
import pandas as pd
from LLm import call_llm





def trace_full_upstream_iterative(req_id, requirements):
    """
    Iteratively trace each input of the given requirement all the way upstream
    until the first requirement (no more inputs).
    Returns dict: {input_symbol: [list of req_ids in lineage]}
    """
    req_index = {req['req_id']: idx for idx, req in enumerate(requirements)}
    if req_id not in req_index:
        return {}

    current_idx = req_index[req_id]
    input_lineages = []

    for inp in requirements[current_idx]['inputs']:
        lineage = []
        stack = [(inp, current_idx)]  # (symbol to trace, current requirement index)

        while stack:
            symbol, idx = stack.pop()

            # search backwards for who produced this symbol
            for r in reversed(requirements[:idx]):
                if symbol in r['outputs'] or symbol in r['inputs']:
                    lineage.append(r['req_id'])
                    # push its inputs to stack to trace further
                    # for new_inp in r['inputs']:
                    #     stack.append((new_inp, req_index[r['req_id']]))
                    break  # stop at first match

        input_lineages.append(lineage)

    return input_lineages



def find_downstream_outputs_only(requirements, start_req_id):
    """
    Trace downstream requirements starting from a given req_id.
    Downstream = any later requirement that uses the current outputs as inputs.
    """

    # Find starting requirement index and outputs
    start_index = None
    start_outputs = []
    for i, req in enumerate(requirements):
        if req["req_id"] == start_req_id:
            start_index = i
            start_outputs = req.get("outputs", [])
            break

    if start_index is None:
        return []  # req_id not found

    downstream = []
    to_trace = set(start_outputs)

    # Walk forward through requirements
    for j in range(start_index + 1, len(requirements)):
        req = requirements[j]
        inputs = set(req.get("inputs", []))

        # If requirement consumes any of the current outputs
        if to_trace & inputs:
            downstream.append(req["req_id"])
            # Expand trace set with this requirement's outputs
            to_trace |= set(req.get("outputs", []))

    return downstream


def reorder_list(reference, output):
    # Keep only items that exist in both, and sort according to reference order
    return [req for req in reference if req in output]


def flatten_list(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def get_upstream_downstream_ech_req(requirements):
    req_ids=[]
    for req in requirements:
        req_ids.append(req['req_id'])
    main_list=[]
    for id in req_ids:
        upstream_ids=[]
        upstream_ids=trace_full_upstream_iterative(id,requirements) 
        # downstream_ids=get_downstream(id,requirements)
        downstream_ids=find_downstream_outputs_only(requirements,id)
        upstream_ids.append(id)
        upstream_ids.append(downstream_ids)
        output=flatten_list(upstream_ids)
            # print(upstream_ids)
        reordered = reorder_list(req_ids, output)
        main_list.append({id:reordered})
    req_flat_dict = {list(d.keys())[0]: list(d.values())[0] for d in main_list}
    return req_flat_dict



def derive_usecase(requirement_text):
    
    prompt=f""" You are given a requirement or context text.  
    Your task is to analyze the text and generate two fields:
    
    - "usecase": A short phrase (3–7 words) describing the functional purpose of the requirement.  
    - "description": A one-line summary explaining what the requirement does(30 words)
    -only one usecase
    -only one description
    
    Return the result strictly in JSON format.
    Strictly JSON OJECT
    STRICTLY JSON OBJECT
    STRICTly JSON OBJECT
    
    Context:
    {requirement_text}
    
    Output:
    {{
      "usecase": "...",
      "description": "..."
    }}"""

    
    return prompt
    
def check_json(t):
    try:
        scenario_json = json.loads(t)
        return scenario_json
    except json.JSONDecodeError as e:
        return "Failed"



def get_llm_retry(prmpt):
    
    max_retries=10   
    for attempt in range(max_retries):
            rep=call_llm(prmpt)
            t=rep.json()['choices'][0]['message']['content']
            out=check_json(t)
            if out == "Failed":
                print(f"⚠️ Attempt {attempt+1}: LLM generation failed, retrying...USECASE")
                continue  # retry again
    
            return out

    return "LLm Failed"




def add_use_case_description(Final_cluster):
    for i in Final_cluster:
        combined_text=" ".join(i['content'])
        prmpt=derive_usecase(combined_text)
        scenario_json=get_llm_retry(prmpt)
        i['use_case']=scenario_json['usecase']
        i['description']=scenario_json['description']
    return Final_cluster


def normalize_req_id(req_id: str) -> str:
    # Remove extra spaces and join parts
    parts = req_id.split()   # split on ANY whitespace, handles multiple spaces
    return "".join(parts)



def get_inputs_from_merged(unique,requirements):
    final=[]
    inputs=[]
    outputs=[]
    for uni in unique:
        uni=normalize_req_id(uni)
        for req in requirements:
            re1=req['req_id']
            re=normalize_req_id(re1)
            if re==uni:
                inputs.append(req['inputs'])
                outputs.append(req['outputs'])
    unique_inputs = list(set(item for sub in inputs for item in sub))
    unique_outputs = list(set(item for sub in outputs for item in sub))

    
    return unique_inputs,unique_outputs

def get_content_from_raw(unique,requirements):
    final=[]
    for uni in unique:
        uni=normalize_req_id(uni)
        for req in requirements:
            re1=req['req_id']
            re=normalize_req_id(re1)
            if re==uni:
                final.append({"req_id":re1,"Content":req['content'],"DiversityExpression":req['diversity_expression']})
    return final

def get_plm_parameters(inputs,outputs,df):
    # inputs=inputs.append(outputs)
    unique = list(set(item for sub in inputs for item in sub))
    flow=df[df['Status']=="PLM Parameters"]['flowtitle'].values
    usecase_inputs = list(set(inputs))
    usecase_outputs = list(set(outputs))
    matches1 = list(set(flow) & set(usecase_inputs))
    matches2 = list(set(flow) & set(usecase_outputs))
    return matches1+matches2

def get_caliber_parameters(inputs,outputs,df):
    # inputs=inputs.append(outputs)
    unique = list(set(item for sub in inputs for item in sub))
    flow=df[df['Status']=="Calibration Parameters"]['flowtitle'].values
    usecase_inputs = list(set(inputs))
    usecase_outputs = list(set(outputs))
    matches1 = list(set(flow) & set(usecase_inputs))
    matches2 = list(set(flow) & set(usecase_outputs))
    return matches1+matches2


def get_External_internal(inputs,df):
    i_map=[]
    for i in inputs:
        match = df[df['flowtitle'] == i]
        if not match.empty:
            if (match['P/C'].iloc[0]=="P") or (match['P/C'].iloc[0]=="C"):
                var = f"{i}[{match['P/C'].iloc[0]}][{match['Signal_and_Frame_Info'].iloc[0]}]"
            else:
                var=f"{i}[{match['P/C'].iloc[0]}]"
                # check if any row matched
        else:
            var = f"{i}[NA]"
        i_map.append(var)
    
    return i_map

def use_case_final(Final_cluster,req_ids_final,flow_df,req_flat_dict,requirements,merged_req):
    usecase=[]
    for use in Final_cluster:
        l=[]
        end_end_list=[]
        new_reordered=[]
        req_id_list=use['req_ids']
        reordered = reorder_list(req_ids_final, req_id_list)
        for j in reordered:
            current_items = req_flat_dict[j]
            
            # check if any item is already in end_end_list
            if any(item in end_end_list for item in current_items):
                # skip (means overlap found → remove from reordered)
                continue
            else:
                end_end_list.extend(current_items)  # append new items
                new_reordered.append(j)  
        # for j in reordered:
        #     end_end_list.append(req_flat_dict[j])
        unique = list(set(end_end_list))
        
        # unique = list(set(item for sub in end_end_list for item in sub))
        unique=reorder_list(req_ids_final, unique)
        final=get_content_from_raw(unique,requirements)
        inputs,outputs=get_inputs_from_merged(unique,merged_req)
        plm_params=get_plm_parameters(inputs,outputs,flow_df)
        caliber_params=get_caliber_parameters(inputs,outputs,flow_df)
        inputs=get_External_internal(inputs,flow_df)
        outputs=get_External_internal(outputs,flow_df)
        use={"usecase":use['use_case'],"Description":use['description'],"Inputs":inputs,"Outputs":outputs,"PLM Parameters":plm_params,"CaliberParameters":caliber_params,"Group":final}
        usecase.append(use)
    return usecase

# import pandas as pd
# flow_df=pd.read_csv("flow.csv")
def get_req(inputfile):
    with open(inputfile,"r") as f:
        req=json.load(f)
    return req

def final_grp(merged_req,Final_clusters,df,inputfile):
    req_ids_final=[re['req_id'] for re in merged_req]
    req_flat_dict=get_upstream_downstream_ech_req(merged_req)
    llm_cluster=add_use_case_description(Final_clusters)
    raw_req=get_req(inputfile)
    final_out=use_case_final(llm_cluster,req_ids_final,df,req_flat_dict,raw_req,merged_req)
    return final_out


