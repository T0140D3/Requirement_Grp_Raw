import requests
import json
import re
def check_cluster():
    
# curl http://apps.calypso.intra.chrysler.com:41000/cmd/start?application=genllm
    url = "http://apps.calypso.intra.chrysler.com:41000/cmd/infos"
    params = {"application": "genllm"}   # query parameters
    response = requests.get(url, params=params)
    def start_cluster():
        url="http://apps.calypso.intra.chrysler.com:41000/cmd/start"
        params = {"application": "genllm"}
        response = requests.get(url, params=params)
        
        
    # 
    if response.json()['state']=="STOP":
        start_cluster()
    
    while True:
        url = "http://apps.calypso.intra.chrysler.com:41000/cmd/infos"
        params = {"application": "genllm"}   # query parameters
        response = requests.get(url, params=params)
        if response.json()['state']=="ONLINE":
            return "True"



def Enrich_content_prompt(requirements_chunk):
    requirements_text = json.dumps(requirements_chunk, indent=2)

    prompt = f"""

You are an expert business analyst specializing in requirement analysis and natural language processing. Your task is to convert technical requirements into clear, understandable natural language descriptions.
You are given multiple requirements in JSON format:
{requirements_text}
INSTRUCTIONS:
1. Read and analyze the provided requirement content thoroughly
2. Identify the core business intent and functionality described
3. Convert any technical variable names, method names, or system identifiers into human-readable terms by understanding their semantic meaning
4. Transform camelCase, PascalCase, or snake_case variables into natural language (e.g., "HornActivationRequest" → "horn activation request", "getUserProfile" → "get user profile", "payment_status_check" → "payment status check")
6. Remove all technical jargon, code syntax, and variable references
7. Use active voice and clear, concise language
8. Ensure the output is understandable to non-technical stakeholders

OUTPUT FORMAT:
Return your response as a JSON object with the following structure:
{{
  "req_id":<"requirement_id>",
  "original_content": "<The exact original requirement text>",
  "natural_language_description": "<Clear, jargon-free description of what the requirement means in business terms>"
}}
STRICTLY JSON object only

"""

    return prompt



def actros_targest_nouns_verbs_prmt(requirements_chunk):
  
  requirements_text = json.dumps(requirements_chunk, indent=2)
  prompt = f"""
You are a requirements analysis expert. Your task is to extract actors, targets, verbs, and nouns from the given requirement content.

You are given multiple requirements in JSON format:
{requirements_text}

STRICT RULES:
1. Analyze ONLY the provided requirement content.
2. Do NOT add any external information or assumptions.
3. Do NOT include variable names, method names, or technical identifiers in any category.
4. Convert technical terms to natural language concepts only.
5. Extract only meaningful business entities and actions.
6. Output must be in strict JSON format.
7. Use lowercase for consistency.

DEFINITIONS:
- actors: who performs the action (users, systems, roles that initiate or perform actions)
- targets: what is being acted upon (objects, systems, data that receive actions)
- verbs: business actions or operations (what is being done)
- nouns: business entities or concepts (things involved in the process)

EXTRACTION GUIDELINES:
- Convert "UserAuthenticationService" → actor: "authentication service", noun: "authentication"
- Convert "validateCredentials()" → verb: "validate", noun: "credentials"
- Convert "HornActivationRequest" → actor: "driver/user", target: "horn system", verb: "activate", noun: "horn"
- Convert "updateAccountBalance" → verb: "update", noun: "account balance"

FORBIDDEN:
- Do not include variable names, class names, method names, technical identifiers
- Do not include programming syntax, camelCase terms, function calls
- Do not add information not present in the requirement

OUTPUT FORMAT (strict JSON, arrays must use brackets []):

  {{
    "req_id": "<requirement_id>",
    "natural_language_description": "<natural_language_description>",
    "actors": ["actor1", "actor2"],
    "targets": ["target1", "target2"],
    "verbs": ["verb1", "verb2"],
    "nouns": ["noun1", "noun2"]
  }}

NOW GENERATE STRICTLY JSON OUTPUT FOR ALL REQUIREMENTS ABOVE. DO NOT ADD ANYTHING ELSE.
STRICTLY JSON OBJECT
STRICTLY JSON OBJECT

"""

  return prompt


def build_prmpt_input(requirements_chunk):
  requirements_text = json.dumps(requirements_chunk, indent=2)
  prompt = f"""
You are an expert requirements parser. Your job is to analyze each requirement and extract ALL variable names as Inputs or Outputs with NO omissions.

Requirements are provided in JSON format:
{requirements_text}

STRICT EXTRACTION RULES:
1. Extract variable names ONLY if they follow mixed case convention (e.g., SpeedRequest, HornControl_Status).  
   - Variables must contain at least one uppercase and one lowercase character.  
   - Discard plain words, lowercase-only terms, or acronyms in full uppercase.  

2. Classification rules:
   - If requirement has an "IF condition", all variables inside IF → Inputs  
   - If requirement has a "THEN" statement, all variables inside THEN → Outputs  
   - If requirement states "system receives X", "when X happens", "check X" → Inputs  
   - If requirement states "system produces Y", "display Y", "update Y", "initialize Y" → Outputs  
   - Always treat initialized variables as Outputs  

3. Inputs and Outputs must follow format: MixedCase or Mixed_Case (example: HornControl_Request).  
   - Do NOT include lowercase-only or uppercase-only words.  
   - Do NOT skip any valid variables.  

OUTPUT FORMAT (STRICT JSON ARRAY):
[
  {{
    "req_id": "<requirement_id>",
    "content": "<original requirement content>",
    "inputs": ["Var1", "Var2"],
    "outputs": ["Var3", "Var4"]
  }},
  ...
]

Do not add explanations or extra text. Output ONLY valid JSON.
"""

  return prompt


def clean(json_str):
    json_str=json_str.replace("\\", " ")
    return json_str
    
def get_json_file(y):
    scenario_json1 = []

    for i in range(len(y)):
        try:
            text = y[i].json()['choices'][0]['message']['content']
        except Exception as e:
            return "Failed"
            

        match_brac = re.search(r"\[.*\]", text, re.DOTALL)
        match_jso = re.search(r"```\n(.*?)\n```", text, re.DOTALL)

        try:
            if match_jso:
                json_str = match_jso.group(1)
                scenario_json = json.loads(json_str)
                scenario_json1.extend(scenario_json)

            elif match_brac:
                json_str = match_brac.group(0)
                json_str = clean(json_str)
                scenario_json = json.loads(json_str)
                scenario_json1.extend(scenario_json)

            else:
                text = "[" + text + "]"
                scenario_json = json.loads(text)
                scenario_json1.extend(scenario_json)

        except json.JSONDecodeError as e:
            return "Failed"
    
    return scenario_json1



def call_llm(prompt):
    # prompt=build_chunk_prompt(chunk)
    url = "http://apps.calypso.intra.chrysler.com:41000/genllm/v1/chat/completions"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    # data = {
    #     "model": "mistral:7b",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": prompt  # <--- prompt here
    #         }
    #     ],
    #     "max_tokens": 3500,
    #     "stream": False
    # }
    data = {
    "model": "mistral:7b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 2500,     # keep lower unless needed
    "temperature": 0,     # randomness
    "top_p": 1.0,           # nucleus sampling
    "stream": False
}
    
    response = requests.post(url, headers=headers, json=data)
    

    # answer_text = response.json()[0]['message']['content']
    # answer_text=response.json()
    
    return response




def generate_content_enrichment(input_file,chunk_size=2):
    
    def load_requirements(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def chunk_requirements(reqs, chunk_size=2):
        for i in range(0, len(reqs), chunk_size):
            yield reqs[i:i+chunk_size]

    requirements = load_requirements(input_file)
    all_results = []

    for chunk in chunk_requirements(requirements, chunk_size):
            prompt=Enrich_content_prompt(chunk)
            # Call your LLM model here with the chunk
            result = call_llm(prompt) 
            all_results.append(result)
    return all_results

def generate_actos_verbs_nouns(requirements,chunk_size=2):
    
    # def load_requirements(file_path):
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         return json.load(f)
    
    def chunk_requirements(reqs, chunk_size=2):
        for i in range(0, len(reqs), chunk_size):
            yield reqs[i:i+chunk_size]


    # requirements = load_requirements(input_file)
    all_results = []
    for chunk in chunk_requirements(requirements, chunk_size):
    
        prompt=actros_targest_nouns_verbs_prmt(chunk)
        # Call your LLM model here with the chunk
        result = call_llm(prompt)
        
        
        all_results.append(result)
    return all_results



def generate_input_files(input_file,chunk_size):
    
    def load_requirements(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def chunk_requirements(reqs, chunk_size=2):
        for i in range(0, len(reqs), chunk_size):
            yield reqs[i:i+chunk_size]

    requirements = load_requirements(input_file)
    all_results = []

    for chunk in chunk_requirements(requirements, chunk_size):
        prompt=build_prmpt_input(chunk)
        # Call your LLM model here with the chunk
        result = call_llm(prompt)
        all_results.append(result)
    return all_results



def merge_content(semantic_data,io_data):
    
    for i in io_data:
        temp = i['req_id'].split(" ")
    
        if len(temp) > 2 and temp[2].strip():  # check if third part exists and not empty
            i['req_id'] = temp[0] + " " + temp[2]
            t = temp[2]
        else:
            if len(temp) > 1:
                i['req_id'] = temp[0] + " " + temp[1]
            else:
                i['req_id'] = temp[0]  # fallback if only one part exists
       
    for i in semantic_data:
        temp = i['req_id'].split(" ")
    
        if len(temp) > 2 and temp[2].strip():  # check if third part exists and not empty
            i['req_id'] = temp[0] + " " + temp[2]
            t = temp[2]
        else:
            if len(temp) > 1:
                i['req_id'] = temp[0] + " " + temp[1]
            else:
                i['req_id'] = temp[0]



# Create a lookup dict for semantic data
    semantic_dict = {item["req_id"]: item for item in semantic_data}
    
    # Merge entries
    merged_data = []
    for io_item in io_data:
        req_id = io_item["req_id"]
        merged_item = {
            "req_id": req_id,
            "inputs": io_item.get("inputs", []),
            "outputs": io_item.get("outputs", [])
        }
    
        # Add semantic info if exists
        semantic_item = semantic_dict.get(req_id, {})
        merged_item.update({"Content":semantic_item.get("natural_language_description"),
            "actors": list(set(semantic_item.get("actors", []))),
            "targets": list(set(semantic_item.get("targets", []))),
            "verbs": list(set(semantic_item.get("verbs", []))),
            "nouns": list(set(semantic_item.get("nouns", [])))
        })
    
        merged_data.append(merged_item)
    return merged_data



def check_llm_output(scenario_json,req_ids):
    if len(scenario_json)!=len(req_ids):
        return "Failed"
    else:
        k=0
        for req in scenario_json:
            req['req_id']=req_ids[k]
            k=k+1
        return scenario_json



def Context_enrichment_run(input_json_file,req_ids, max_retries=5):
    for attempt in range(max_retries):
        y = generate_content_enrichment(input_json_file, chunk_size=2)
        y=get_json_file(y)
        if y == "Failed":
            # print(f"⚠️ Attempt {attempt+1}: LLM generation failed, retrying...")
            continue  # retry again

        flag = check_llm_output(y, req_ids)
        if flag == "Failed":
            continue 

    
        return flag

    return "Failed"



def Inputs_outputs_gen(input_json_file,req_ids,max_retries=5):
    for attempt in range(max_retries):
        
        y=generate_input_files(input_json_file,chunk_size=2)
        out=get_json_file(y)
        if out == "Failed":
            # print(f"⚠️ Attempt {attempt+1}: LLM generation failed, retrying...")
            continue  # retry again

        flag = check_llm_output(out, req_ids)
        if flag == "Failed":
            continue 

    
        return flag

    return "Failed"


def Context_verbs_nouns_run(input_json_file,req_ids,max_retries=5):
    for attempt in range(max_retries):
        y=generate_actos_verbs_nouns(input_json_file,chunk_size=2)
        out=get_json_file(y)
        if out == "Failed":
            # print(f"⚠️ Attempt {attempt+1}: LLM generation failed, retrying...")
            continue  # retry again

        flag = check_llm_output(out, req_ids)
        if flag == "Failed":
            continue 

    
        return flag

    return "Failed"


def get_requriements_data(inputfile):
    with open(inputfile,"r") as f:
        raw_req=json.load(f)
    req_ids=[req['req_id'] for req in raw_req]
    return req_ids


def calling_llm(inputfile):
    req_ids=get_requriements_data(inputfile)
    max_retries=10
    Context_Enrich=Context_enrichment_run(inputfile,req_ids,max_retries)
    Nouns_verbs=Context_verbs_nouns_run(Context_Enrich,req_ids,max_retries)
    inputs_outputs=Inputs_outputs_gen(inputfile,req_ids,max_retries)
    final=merge_content(Nouns_verbs,inputs_outputs)
    return final