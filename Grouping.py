from sentence_transformers import SentenceTransformer

from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import DBSCAN
import community 
# Download resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def tune_dbscan_eps(embeddings, min_samples, alpha):
    # Step 1: Compute pairwise distances
    dists = cosine_distances(embeddings)
    
    # Step 2: Take k-distances (distance to min_samples-th neighbor)
    k = min_samples
    sorted_dists = np.sort(np.partition(dists, k, axis=1)[:, k])
    
    # Step 3: Candidate eps values from percentiles
    candidate_eps = np.percentile(sorted_dists, np.arange(2, 20, 1))  # 2% to 20%
    
    best_score, best_eps, best_labels = -np.inf, None, None
    dist_matrix = cosine_distances(embeddings)
    dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    # Step 4: Try each eps
    for i in range(1,15):
        eps = np.percentile(dists, i)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = db.fit_predict(embeddings)
        num_clusters = len(set(labels))
        num_outliers = np.sum(labels == -1)
        # Score: maximize clusters, penalize outliers
        score = num_clusters - alpha * (num_outliers / len(labels))
        
        if score > best_score:
            best_score, best_eps, best_labels = score, eps, labels
    
    return best_eps, best_labels
# eps,labels=tune_dbscan_eps(X, min_samples=2, alpha=0.5)


def DB_SCAN_ALGO(X):
    min_samples=2
    eps,labels=tune_dbscan_eps(X, min_samples=2, alpha=0.5)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(X)
    uni=np.unique(labels)
    clusters={}
    for i in uni:
        indices_cluster_0 = list(np.where(labels == i)[0])
        # clusters.append({i:list(indices_cluster_0)})
        clusters.setdefault(i,[]).append(indices_cluster_0)
    return clusters
    






def preprocess(text):
    words = nltk.word_tokenize(text)
    filtered_words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    lemmas = [lemmatizer.lemmatize(w, pos="v") for w in filtered_words]
    return " ".join(lemmas)

def get_merged(merged_req):
    processed_texts=[preprocess(req['full_Text']) for req in merged_req]
    return processed_texts

def flatten_requirement(content_lists):
        return " ".join(["".join(sublist) for sublist in content_lists])

def extract_clean_text(merged_req):
    requirement_texts=[]
    for i in merged_req:
        temp=[]
        temp.append(i['Content'])
        temp.append(i['targets'])
        temp.append(i['actors'])
        temp.append(i['inputs'])
        temp.append(i['outputs'])
        temp.append(i['verbs'])
        temp.append(i['nouns'])
        # temp.append(i['key_phrase'])
        requirement_texts.append(temp)
   

    req_texts = [flatten_requirement(lists) for lists in requirement_texts]
    k=0
    for req in merged_req:
        req['full_Text']=req_texts[k]
        k=k+1
    req_ids_final=[re['req_id'] for re in merged_req]
    return merged_req,req_ids_final


def create_embeddings(merged_req):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([req['full_Text'] for req in merged_req])
    return embeddings



# def build_graph(embeddings):
#     sim_matrix = cosine_similarity(embeddings)

# # Build graph
#     G = nx.Graph()
#     n = len(sim_matrix)
#     for i in range(n):
#         for j in range(i+1, n):
#             if sim_matrix[i, j] > 0.70:  # similarity threshold
#                 G.add_edge(i, j, weight=sim_matrix[i, j])
    
#     # Louvain community detection
#     partition = community.best_partition(G, weight='weight')
    
#     # Map requirement IDs to clusters
#     clusters = {}
#     for node, cluster_id in partition.items():
#         clusters.setdefault(cluster_id, []).append(node)
#     return clusters



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


def refine_clusters(clusters,req_ids_final,merged_req):
     Final_cluster=[]  
     for c,k in clusters.items():
                req_id=[req_ids_final[j] for j in k[0]]
                req_id=reorder_list(req_ids_final,req_id)
                content=[]
                for rid in req_id:
                    for i in merged_req:
                        if i['req_id']==rid:
                            content.append(i['full_Text'])
                Final_cluster.append({"id":c,"req_ids":req_id,"content":content})
     return Final_cluster
    
vectorizer = TfidfVectorizer(stop_words="english")

def group_logic(merged_req):
    merged_req,req_ids_final=extract_clean_text(merged_req)
    preprocessed_txt=get_merged(merged_req)
    X = vectorizer.fit_transform(preprocessed_txt)
    clusters=DB_SCAN_ALGO(X)
    # embeddings=create_embeddings(merged_req)
    # clusters=build_graph(embeddings)
    final_clusters=refine_clusters(clusters,req_ids_final,merged_req)
    return final_clusters

    

        
