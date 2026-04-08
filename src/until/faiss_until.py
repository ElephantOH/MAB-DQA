import os
CURRENT_FILE_PATH = os.path.abspath(__file__)

import faiss
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm


class FaissIndexHandler:
    """FAISS 索引处理器，用于文档嵌入检索（支持 ColPali/ColBERT 模型）"""
    def __init__(
            self,
            dataset,
            index_type: str = "flatip",
            embed_dim: int = 128,
        ):
  
        self.embed_dim = embed_dim
        self.index_type = index_type
        self.index_path = dataset.index_path
        self.index = self._create_index()
        self.token_index_to_page_uid = []
        self.all_embeds = None

    def _create_index(self):
        """私有方法：根据索引类型创建 FAISS 索引"""
        quantizer = faiss.IndexFlatIP(self.embed_dim)
        
        if self.index_type == "flatip":
            return quantizer
        elif self.index_type == "ivfflat":
            ncentroids = 1024
            return faiss.IndexIVFFlat(quantizer, self.embed_dim, ncentroids)
        elif self.index_type == "ivfpq":
            nlist, sub_quantizers = 100, 8
            return faiss.IndexIVFPQ(quantizer, self.embed_dim, nlist, sub_quantizers, 8)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def build_index(
        self, 
        document_id_to_embeds: dict, 
        model_type: str, 
        document_id_to_lengths: dict = None
    ):
        """
        从文档嵌入向量构建 FAISS 索引
        Args:
            document_id_to_embeds: 字典 {文档ID: 嵌入张量}
            model_type: 模型类型（colpali/colbert）
            document_id_to_lengths: ColBERT 专用参数（页面token长度）
        """
        all_token_embeds = []
        self.token_index_to_page_uid = []

        if model_type == "colpali":
            total_docs = len(document_id_to_embeds)
            for document_id, document_embed in tqdm(document_id_to_embeds.items(), total=total_docs):
                for page_id in range(len(document_embed)):
                    page_embed = document_embed[page_id].view(-1, self.embed_dim)
                    all_token_embeds.append(page_embed)
                    page_uid = f"{document_id}_page{page_id}"
                    self.token_index_to_page_uid.extend([page_uid] * page_embed.shape[0])

        elif model_type == "colbert":
            if document_id_to_lengths is None:
                raise ValueError("ColBERT 模型必须传入 document_id_to_lengths 参数")
            total_docs = len(document_id_to_embeds)
            for document_id, document_embed in tqdm(document_id_to_embeds.items(), total=total_docs):
                document_lengths = document_id_to_lengths[document_id]
                all_token_embeds.append(document_embed)
                for page_id, page_length in enumerate(document_lengths):
                    page_uid = f"{document_id}_page{page_id}"
                    self.token_index_to_page_uid.extend([page_uid] * page_length.item())
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.all_embeds = torch.cat(all_token_embeds, dim=0).float().numpy()

        print(f"[Info] [{CURRENT_FILE_PATH}] Training index with vectors of shape {self.all_embeds.shape}.")
        self.index.train(self.all_embeds)
        self.index.add(self.all_embeds)
        print(f"[Info] [{CURRENT_FILE_PATH}] FAISS index built successfully.")

    def save_index(
            self,
            dataset_name,
            mission_key,
            model_type,
            is_clip=False,
        ):
        """将 FAISS 索引保存到本地"""

        self.index_path.mkdir(exist_ok=True)
        filename = f"index[dataset_{dataset_name}][mission_{mission_key}][model_{model_type}]{"[Clip]" if is_clip else ""}.bin"
        save_path = str(self.output_dir / filename)
        faiss.write_index(self.index, save_path)
        print(f"[Info] [{CURRENT_FILE_PATH}] Index saved to: {save_path}.")

    def load_index(
            self,
            dataset_name,
            mission_key,
            model_type,
            is_clip=False,
        ):
        """从本地加载 FAISS 索引"""
        filename = f"index[dataset_{dataset_name}][mission_{mission_key}][model_{model_type}]{"[Clip]" if is_clip else ""}.bin"
        load_path = str(self.index_path / filename)
        self.index = faiss.read_index(load_path)
        print(f"[Info] [{CURRENT_FILE_PATH}] Index loaded from: {load_path}.")

    def search(self, query_embed: np.ndarray, top_k: int = 10):
        """
        在 FAISS 索引中检索最近邻向量
        Args:
            query_embed: 查询嵌入数组（形状：[token数量, 向量维度]）
            top_k: 返回的最相似结果数量
        Returns:
            D: 相似度距离/分数, I: 匹配的token索引
        """
        return self.index.search(query_embed, top_k)

    def compute_maxsim_scores(self, query_embed: np.ndarray, top_k: int = 10):
        """
        计算文档页面 MaxSim 分数（ColBERT 标准排序逻辑）
        Args:
            query_embed: 查询嵌入数组
            top_k: 返回Top-K排序结果
        Returns:
            按分数降序排列的 (页面唯一标识, 分数) 列表
        """
        similarity_scores, token_indices = self.search(query_embed, top_k)
        page_total_scores = {}

        # 遍历每个查询token，计算MaxSim分数
        for query_idx in range(query_embed.shape[0]):
            current_token_page_scores = {}
            for neighbor_idx in range(top_k):
                matched_token_idx = token_indices[query_idx, neighbor_idx]
                page_uid = self.token_index_to_page_uid[matched_token_idx]
                # 计算余弦相似度
                score = (query_embed[query_idx] * self.all_embeds[matched_token_idx]).sum()
                
                # 保留单个页面的最高分数
                if page_uid not in current_token_page_scores or score > current_token_page_scores[page_uid]:
                    current_token_page_scores[page_uid] = score

            # 累加所有查询token的分数
            for uid, single_score in current_token_page_scores.items():
                page_total_scores[uid] = page_total_scores.get(uid, 0) + single_score

        # 按总分降序排序，返回Top-K结果
        sorted_pages = sorted(page_total_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:top_k]



from pathlib import Path



def main():
  
    dataset = None
    document_id_to_embeds = dataset.load_all_embeddings()
    document_id_to_lengths = None
    
    index_handler = FaissIndexHandler(
        embed_dim=128,
        index_type="args.faiss_index_type",
        output_dir="args.output_dir",
    )

    index_handler.build_index(document_id_to_embeds, model_type="colpali")

    index_handler.save_index()

    import numpy as np
    example_query_embed = np.random.randn(20, 128).astype(np.float32)
    top_pages = index_handler.compute_maxsim_scores(example_query_embed, top_k=10)

    print("Top-K page candidates:")
    for uid, score in top_pages:
        print(f"{uid}: {score:.4f}")

if __name__ == "__main__":
    main()
