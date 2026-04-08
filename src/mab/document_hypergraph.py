import os
import torch
import numpy as np
from typing import List, Dict, Set
from src.until.page_similarity_graph import construct_page_similarity_graph


class DocumentHypergraph:
    def __init__(self):
        self.page_similarity_graph = {}
        self.query_specific_hypergraph = {}
        self.graph = {}

    def construct_page_similarity_graph(
        self,
        page_embeds,
        threshold: float = 0.7,
        k_value: int = 5,
        similarity_measure: str = "cosine"
    ) -> Dict[int, List[int]]:
        self.page_similarity_graph = construct_page_similarity_graph(
            page_embeds=page_embeds,
            threshold=threshold,
            k_value=k_value,
            similarity_measure=similarity_measure
        )

    def _get_specific_page(
        self,
        basis_pages: List[int],
        target_pages: List[int]
    ) -> Set[int]:
        set_basis, set_target = set(basis_pages), set(target_pages)
        new_pages = set_target - set_basis
        improved_pages = {
            p for p in set_basis & set_target
            if target_pages.index(p) < basis_pages.index(p)
        }
        return new_pages | improved_pages

    def construct_query_specific_hypergraph(
        self,
        bandit,
        vlm,
        dataset,
        top_page_indices: List[List[int]],
        top_page_scores: List[List[float]],
        queries: List[str],
        col_score_dict: Dict[int, float],
        sample: Dict,
        best_first_hit_score: int = 20
    ):

        document_hypergraph = {}
        extra_pages = set()
        default_idx = len(top_page_indices) - 1
        bf_query_idx, bf_vlm_score, bf_score, bf_page = default_idx, -999, -999, 0

        def _load_node(
            page_num: int,
            query_idx: int,
            use_vlm: bool = True,
            first_hit: int = 1,
            multi_hit: int = 1
        ):
            nonlocal bf_query_idx, bf_vlm_score, bf_score, bf_page
            if page_num in document_hypergraph:
                node = document_hypergraph[page_num]
                node["hits"] += multi_hit
                cb_score = bandit.sample(node.get("query_idx", set()))
                node["score"] = bandit._compute_score(
                    col_score=node["col_score"],
                    vlm_score=node["vlm_score"],
                    cb_score=0.5,
                    hits=node["hits"],
                )
                if query_idx != default_idx and query_idx not in node["query_idx"]:
                    node["query_idx"].add(query_idx)
                    node["query"] += f",{queries[query_idx]}"
            else:
                vlm_score = -1
                if use_vlm:
                    vlm_score = bandit.query_vlm_relevance(
                        vlm=vlm,
                        dataset=dataset,
                        sample=sample,
                        page=page_num,
                        priori=queries[query_idx],
                    )
                    bandit.update({query_idx}, vlm_score)

                specific_pages = self._get_specific_page(
                    top_page_indices[default_idx],
                    top_page_indices[query_idx]
                ) if query_idx != default_idx else set(top_page_indices[default_idx])
                neighbors = specific_pages | set(self.page_similarity_graph.get(page_num, []))
                extra_pages.update(neighbors - specific_pages)

                cb_score = bandit.sample({query_idx})
                document_hypergraph[page_num] = {
                    "query_idx": {query_idx},
                    "query": queries[query_idx],
                    "col_score": col_score_dict[page_num],
                    "vlm_score": vlm_score,
                    "hits": first_hit,
                    "cb_score": cb_score,
                    "score": bandit._compute_score(col_score_dict[page_num], vlm_score, 0.5, first_hit),
                    "neighbor": neighbors,
                }

                if use_vlm:
                    print(f"{queries[query_idx]} [page:{page_num}]: \n - vlm: {document_hypergraph[page_num]["vlm_score"]}\n - col: {document_hypergraph[page_num]["col_score"]}")    
                    if (vlm_score > bf_vlm_score or (vlm_score == bf_vlm_score and document_hypergraph[page_num]["score"] > bf_score + 0.0001)):
                        bf_query_idx, bf_vlm_score, bf_score, bf_page = query_idx, vlm_score, document_hypergraph[page_num]["score"], page_num

        for query_idx in reversed(range(len(top_page_indices))):
            _load_node(
                page_num=top_page_indices[query_idx][0],
                query_idx=query_idx,
                use_vlm=True,
                first_hit=best_first_hit_score,
            )

        if bf_page in document_hypergraph:
            print(f" [DEBUG]  best_first_queries: {queries[bf_query_idx]}")
            sample["best_first_queries"] = queries[bf_query_idx]
            document_hypergraph[bf_page]["hits"] += best_first_hit_score
            cb_score = bandit.sample(document_hypergraph[bf_page]["query_idx"])
            document_hypergraph[bf_page]["score"] = bandit._compute_score(
                col_score=document_hypergraph[bf_page]["col_score"],
                vlm_score=document_hypergraph[bf_page]["vlm_score"],
                cb_score=cb_score,
                hits=document_hypergraph[bf_page]["hits"],
            )

        for page_num in top_page_indices[bf_query_idx][:10]:
            _load_node(
                page_num=page_num,
                query_idx=bf_query_idx,
                use_vlm=False,
                first_hit=best_first_hit_score
            )

        for query_idx in range(len(top_page_indices)):
            for page_num in top_page_indices[query_idx]:
                _load_node(
                    page_num=page_num,
                    query_idx=query_idx,
                    use_vlm=False,
                    first_hit=1
                )

        for page_num in extra_pages:
            if page_num not in document_hypergraph:
                document_hypergraph[page_num] = {
                    "query_idx": set(),
                    "query": queries[default_idx],
                    "col_score": col_score_dict[page_num],
                    "hits": 1,
                    "cb_score": bandit.sample({default_idx}),
                    "neighbor": set(),
                }

        self.query_specific_hypergraph = document_hypergraph

    def _evaluate_rag_one_sample(self, gt, pred, top_k=[1, 3, 5]):
        metrics = 0
        len_gt = len(gt)
        for k in top_k:
            cur_pred = pred[:k]
            intersect = len(set(cur_pred) & set(gt))
            metrics += intersect / len_gt * 100.0
            metrics += intersect / len(cur_pred) * 100.0
        return int(metrics)

    def _debug(self, dataset, sample, queries, scores):
        if "evidence_pages" not in sample:
            print("[Warning]: No ground_truth.")
            return
        ground_truth = sample["evidence_pages"]
        if isinstance(ground_truth, str):
            try:
                ground_truth = [int(page.strip()) for page in
                                ground_truth.strip('[]').split(',')] if ground_truth != "[]" else []
            except (ValueError, AttributeError):
                print("Error parsing evidence_pages string")
                ground_truth = []
        else:
            ground_truth = list(ground_truth)

        if ground_truth != [] and len(ground_truth) > 0:

            if dataset.dataset_name == "MMLongBench":
                ground_truth = [g - 1 if g > 0 else g for g in ground_truth]
            else:
                ground_truth = [g if g > 0 else g for g in ground_truth]

            query_top = torch.topk(torch.tensor(np.array(scores)), min(10, len(scores[0])), dim=-1)
            query_top_indices = query_top.indices.tolist()
            query_top_scores = query_top.values.tolist()

            metrics_list = []
            for i, top_indices in enumerate(query_top_indices):
                metrics = self._evaluate_rag_one_sample(ground_truth, top_indices)
                metrics_list.append((metrics, queries[i], query_top_indices[i], query_top_scores[i]))
            metrics_list.sort(key=lambda x: x[0], reverse=True)

            sample["sorted_queries"] = {query: metric for metric, query, _, _ in metrics_list}

            print("\n", "#" * 30)
            print("[DEBUG] Ground Truth: ", ground_truth)
            result = "".join([f" [{query}, {metric}]\n" for metric, query, _, _ in metrics_list])
            print("[DEBUG] True top Indices and Scores: \n" + result)    
            print("#" * 30, "\n")

    def clean_up_page_similarity_graph(self):
        self.page_similarity_graph = {}

    def clean_up_query_specific_hypergraph(self):
        self.query_specific_hypergraph = {}

