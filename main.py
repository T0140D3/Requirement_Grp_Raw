import os
import json
import pandas as pd
from LLm import check_cluster,calling_llm
from Grouping import group_logic
from final_grouping import final_grp
from raw_data_parser import run_cli
from Requirements_Flows_Extractor_UI import flows_Extraction

input_word_file="horn.docx"
outputfile="requirements1.json"

DCI_document="DCI_Global_Brain_25Q1.xlsx"

run_cli(input_word_file,outputfile)
flows_df=flows_Extraction(input_word_file,DCI_document)

t=check_cluster()
print(t)

if t:
    inputfile="requirements1.json"
    # df=pd.read_csv("flow.csv")
    print("LLM is working on .............")
    llm_out=calling_llm(inputfile)
    print("Grouping.......")
    group_info=group_logic(llm_out)
    print("Finalising....")
    final_out=final_grp(llm_out,group_info,flows_df,inputfile)
    
    with open("Requirements_grouping.json","w") as f:
        json.dump(final_out,f,indent=4)
       
  
    