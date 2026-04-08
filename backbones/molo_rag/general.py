import pytz 
import datetime 
import time 
import numpy as np
import torch


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def distnace_similarity(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    if distance < 1e-9:
        return 1.0 
    return np.exp(-distance)


def similarity_func(vec1, vec2, metric="cosine"):
    assert metric in ["cosine", "distance"]
    if metric == "cosine": 
        return cosine_similarity(vec1, vec2)
    
    return distnace_similarity(vec1, vec2)


def compute_embed_similarity(doc1_embed, doc2_embed, sim_func="distance"):
    # 使用PyTorch计算均值
    if torch.is_tensor(doc1_embed):
        doc1_avg = torch.mean(doc1_embed, dim=0)
    else:
        doc1_avg = np.mean(doc1_embed, axis=0)

    if torch.is_tensor(doc2_embed):
        doc2_avg = torch.mean(doc2_embed, dim=0)
    else:
        doc2_avg = np.mean(doc2_embed, axis=0)

    # 确保两个平均值都是相同类型（NumPy数组）
    if torch.is_tensor(doc1_avg):
        doc1_avg = doc1_avg.detach().cpu().float().numpy()
    if torch.is_tensor(doc2_avg):
        doc2_avg = doc2_avg.detach().cpu().float().numpy()

    return similarity_func(doc1_avg, doc2_avg, sim_func)
