import re
import numpy as np
from scipy.stats import beta
from src.prompt.prompt_mab import PROMPTS
from scipy.stats import beta
from typing import List, Dict, Tuple


class ThompsonSamplingBandit:
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.arms: Dict[int, Tuple[float, float]] = {}  # {query_id: (alpha, beta)}
        self.alpha_val = 0.8
        self.beta_val = 0.01
        self.lambda_val = 1

    def add_arm(self, query_id: int):
        if query_id not in self.arms:
            self.arms[query_id] = (self.alpha_prior, self.beta_prior)

    def update(self, query_ids: List[int], reward: float):
        for qid in query_ids:
            self.add_arm(qid)
            alpha_val, beta_val = self.arms[qid]
            self.arms[qid] = (alpha_val + reward, beta_val + (1 - reward))

    def sample(self, query_ids: List[int]) -> float:
        if not query_ids:
            return 0.0
        samples = []
        for qid in query_ids:
            if qid in self.arms:
                alpha_val, beta_val = self.arms[qid]
                samples.append(beta.rvs(alpha_val, beta_val))
            else:
                samples.append(self.alpha_prior / (self.alpha_prior + self.beta_prior))
        return float(np.mean(samples))
    
    def clean_all_arms(self):
        self.arms = {}

    def _compute_score(
        self,
        col_score: float,
        vlm_score: float,
        cb_score: float,
        hits: int,
    ) -> float:
        _vlm = max(vlm_score, 0.0)
        return (1 - self.alpha_val) * col_score + self.alpha_val * _vlm + self.beta_val * (self.lambda_val * hits + (1 - self.lambda_val) * cb_score)
    

    def query_vlm_relevance(
        self,
        vlm,
        dataset,
        sample,
        page,
        priori=None,
    ):
        if priori is not None and priori != "" and priori != sample["initial_query"] and priori != sample["basis_queries"]:
            priori = priori
        else:
            priori = None
        try:
            contents = dataset.extract_page_contents(
                sample=sample,
                page=page,
                load_contents=True,
                save_contents=True,
                clip_border=True,
            )
            image = contents[0].image

            if dataset.dataset_name == "MMLongBench":
                if priori is None:
                    prompt = PROMPTS["retrieval"](sample["initial_query"])
                else:
                    prompt = PROMPTS["conditional_retrieval"](priori, sample["initial_query"])
            else:
                prompt = PROMPTS["retrieval_detailed"](sample["initial_query"])
            response, _ = vlm.predict(
                question=prompt,
                texts=None,
                images=[image],
                max_new_tokens=16,
            )
            score_match = re.search(r'[1-5]', response)
            if score_match:
                relevance_score = int(score_match.group(0))
            else:
                relevance_score = 3
        except Exception as e:
            print(f"[Error] Error in VLM for {page}; Exception {e}. Returning Default of 1.")
            relevance_score = 1
        normalized_score = (relevance_score - 1.0) / 4.0
        return normalized_score


    def mab_retrieval(
        self,
        document_hypergraph,
        vlm,
        dataset,
        sample,
        m: int = 5,
        k: int = 10,
        b: int = 2,
    ) -> Tuple[List[int], List[float]]:

        graph = document_hypergraph.query_specific_hypergraph
        def _get_top_k(select: str = "all", limit: int = k) -> List[int]:
            nodes = []
            for page_num, info in graph.items():
                cb_score = self.sample(info.get("query_idx", set()))
                info["cb_score"] = cb_score
                if select == "all" and "vlm_score" in info and info["vlm_score"] is not None:
                    info["score"] = self._compute_score(
                        col_score=info["col_score"],
                        vlm_score=info.get("vlm_score", -1.0),
                        cb_score=cb_score,
                        hits=info["hits"]
                    )
                    nodes.append((page_num, info["score"]))
                elif select == "unvisited" and ("vlm_score" not in info or info["vlm_score"] is None or info["vlm_score"] < 0):
                    info["score"] = self._compute_score(
                        col_score=info["col_score"],
                        vlm_score=info.get("vlm_score", -1.0),
                        cb_score=cb_score,
                        hits=info["hits"]
                    )
                    nodes.append((page_num, info["score"]))
            nodes.sort(key=lambda x: x[1], reverse=True)
            return [p for p, _ in nodes[:limit]]

        def _get_neighbor_top_k(neighbors: List[int], select: str = "all", limit: int = k) -> List[int]:
            nodes = []
            for page_num in neighbors:
                info = graph[page_num]
                cb_score = self.sample(info.get("query_idx", set()))
                info["cb_score"] = cb_score
                if select == "unvisited" and ("vlm_score" not in info or info["vlm_score"] is None or info["vlm_score"] < 0):
                    info["score"] = self._compute_score(
                        col_score=info["col_score"],
                        vlm_score=info.get("vlm_score", -1.0),
                        cb_score=cb_score,
                        hits=info["hits"]
                    )
                    nodes.append((page_num, info["score"]))
            nodes.sort(key=lambda x: x[1], reverse=True)
            return [p for p, _ in nodes[:limit]]

        for round_num in range(b):
            print(f"=== Round {round_num + 1} ===")
            top_m = _get_top_k(limit=m)
            print(f"Selected Top-{len(top_m)} Nodes: {top_m}")
            any_op = False

            for page_num in top_m:
                node = graph[page_num]
                if "vlm_score" not in node or node["vlm_score"] < 0:
                    vlm_score = self.query_vlm_relevance(
                        vlm=vlm,
                        dataset=dataset,
                        sample=sample,
                        page=page_num,
                        priori=node["query"]
                    )
                    self.update(node.get("query_idx", set()), vlm_score)
                    cb_score = self.sample(node.get("query_idx", set()))
                    node["vlm_score"] = vlm_score
                    node["cb_score"] = cb_score
                    node["score"] = self._compute_score(
                        col_score=node["col_score"],
                        vlm_score=vlm_score,
                        cb_score=cb_score,
                        hits=node["hits"]
                    )
                    print(f"[Info] Computing node {page_num}'s scores, col_score: {node['col_score']}, "
                        f"vlm_score: {vlm_score}, hits: {node['hits']}, cb_score: {cb_score:.4f}, score: {node['score']:.4f}.")
                    any_op = True
                else:
                    candidates = _get_neighbor_top_k(node.get("neighbor", []), select="unvisited", limit=k)
                    for neighbor in candidates:
                        n_node = graph[neighbor]
                        n_penlty = 0.5
                        if "vlm_score" not in n_node or n_node["vlm_score"] < 0:
                            vlm_score = self.query_vlm_relevance(
                                vlm=vlm,
                                dataset=dataset,
                                sample=sample,
                                page=neighbor,
                                priori=n_node["query"]
                            )
                            self.update(n_node.get("query_idx", set()), vlm_score)
                            cb_score = self.sample(n_node.get("query_idx", set()))
                            n_node["vlm_score"] = vlm_score * n_penlty
                            n_node["cb_score"] = cb_score
                            n_node["score"] = self._compute_score(
                                col_score=n_node["col_score"],
                                vlm_score=vlm_score,
                                cb_score=cb_score,
                                hits=n_node["hits"]
                            )
                            print(f"[Info] Computing node {neighbor}'s scores, col_score: {n_node['col_score']}, "
                                f"vlm_score: {vlm_score}, hits: {n_node['hits']}, cb_score: {cb_score:.4f}, score: {n_node['score']:.4f}.")
                            any_op = True
                            break

            if not any_op:
                print("No Operations Possible, Exiting.")
                break

        final = []
        for page_num, info in graph.items():
            if "score" in info and info["score"] is not None:
                cb_score = self.sample(info.get("query_idx", set()))
                if "vlm_score" in info and info["vlm_score"] is not None:
                    info["score"] = self._compute_score(
                        col_score=info["col_score"],
                        vlm_score=info["vlm_score"],
                        cb_score=cb_score,
                        hits=info["hits"]
                    )
                final.append((page_num, info["score"]))
        final.sort(key=lambda x: x[1], reverse=True)

        top_k_nodes = [p for p, _ in final[:k]]
        top_k_scores = [s for _, s in final[:k]]
        print(f"Final Top-{k} Nodes: {top_k_nodes}")

        return top_k_nodes, top_k_scores

    
