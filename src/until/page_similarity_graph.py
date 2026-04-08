import numpy as np
import torch
from collections import defaultdict


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def distnace_similarity(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    if distance < 1e-9:
        return 1.0 
    return np.exp(-distance)

def similarity_func(vec1, vec2, similarity_measure="cosine"):
    assert similarity_measure in ["cosine", "distance"]
    if similarity_measure == "cosine": 
        return cosine_similarity(vec1, vec2)
    return distnace_similarity(vec1, vec2)

def compute_embed_similarity(page1_embed, page2_embed, similarity_measure="distance"):
    if torch.is_tensor(page1_embed):
        page1_avg = torch.mean(page1_embed, dim=0)
    else:
        page1_avg = np.mean(page1_embed, axis=0)
    if torch.is_tensor(page2_embed):
        page2_avg = torch.mean(page2_embed, dim=0)
    else:
        page2_avg = np.mean(page2_embed, axis=0)
    if torch.is_tensor(page1_avg):
        page1_avg = page1_avg.detach().cpu().float().numpy()
    if torch.is_tensor(page2_avg):
        page2_avg = page2_avg.detach().cpu().float().numpy()
    return similarity_func(page1_avg, page2_avg, similarity_measure)

def construct_page_similarity_graph(
        page_embeds,
        threshold=0.7,
        k_value=5,
        similarity_measure="cosine"
    ):
    n_pages, _, _ = page_embeds.shape
    edges = []
    similarity_matrix = np.zeros((n_pages, n_pages))
    
    for i in range(n_pages):
        for j in range(i + 1, n_pages):
            page1_embed, page2_embed = page_embeds[i], page_embeds[j]
            sim_score = compute_embed_similarity(page1_embed, page2_embed, similarity_measure=similarity_measure)
            similarity_matrix[i][j] = sim_score
            similarity_matrix[j][i] = sim_score
            
    for i in range(n_pages):
        similarity_scores = similarity_matrix[i]
        top_k_indices = np.argsort(similarity_scores)[::-1][:k_value]

        for j in top_k_indices:
            if similarity_scores[j] >= threshold:
                edges.append((i, j))

    page_similarity_graph = defaultdict(list)
    for u, v in edges:
        page_similarity_graph[int(u)].append(int(v))
        page_similarity_graph[int(v)].append(int(u))

    page_similarity_graph = {k: list(set(v)) for k, v in page_similarity_graph.items()}
    return page_similarity_graph