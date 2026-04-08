import torch
import numpy as np
from typing import List, Optional, Union

def get_torch_device(
        device: str = "auto"
    ) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device

def late_interaction(
        page_embeds: Union[torch.Tensor, List[torch.Tensor]],
        query_embeds: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
        result_to_numpy: bool = True,
    ):
    device = device or get_torch_device("auto")
    if len(query_embeds) == 0:
        raise ValueError("No queries provided")
    if len(page_embeds) == 0:
        raise ValueError("No passages provided")
    scores_list: List[torch.Tensor] = []
    for i in range(0, len(query_embeds), batch_size):
        scores_batch = []
        query_batch = torch.nn.utils.rnn.pad_sequence(query_embeds[i: i + batch_size], batch_first=True, padding_value=0).to(
            device
        )
        for j in range(0, len(page_embeds), batch_size):
            page_batch = torch.nn.utils.rnn.pad_sequence(
                page_embeds[j: j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            scores_batch.append(torch.einsum("bnd,csd->bcns", query_batch, page_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)
    scores = torch.cat(scores_list, dim=0)
    assert scores.shape[0] == len(query_embeds), f"[Info] Expected {len(query_embeds)} Scores, Got {scores.shape[0]}."
    scores = scores.to(torch.float32)
    normalized_scores = []
    for i in range(scores.shape[0]):
        row = scores[i, :]
        min_val = torch.min(row)
        max_val = torch.max(row)
        if max_val - min_val > 0:
            normalized_row = (row - min_val) / (max_val - min_val)
        else:
            normalized_row = torch.zeros_like(row)
        normalized_scores.append(normalized_row)
    normalized_scores = torch.stack(normalized_scores, dim=0)
    max_scores_per_image = torch.max(normalized_scores, dim=0)[0]
    colpali_score_dict = {img_idx: float(max_scores_per_image[img_idx]) for img_idx in range(max_scores_per_image.shape[0])}

    if result_to_numpy:
        scores = scores.to(torch.float32).cpu().numpy()

    return scores, colpali_score_dict

def get_top_page(scores, top_k=10):
    top_page = torch.topk(torch.tensor(np.array(scores)), min(top_k, len(scores[0])), dim=-1)
    return top_page