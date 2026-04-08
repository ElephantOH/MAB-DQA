from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from backbones.colpali.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
            qs: Union[torch.Tensor, List[torch.Tensor]],
            ps: Union[torch.Tensor, List[torch.Tensor]],
            batch_size: int = 128,
            device: Optional[Union[str, torch.device]] = None,
            to_np: bool = True,
    ) -> torch.Tensor:
        device = device or get_torch_device("auto")
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        scores_list: List[torch.Tensor] = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i: i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j: j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        scores = scores.to(torch.float32)
        if to_np:
            scores = scores.to(torch.float32).cpu().numpy()
        return scores

    @staticmethod
    def score_mm_vector(
            queries_embed: Union[torch.Tensor, List[torch.Tensor]],
            images_embed: Union[torch.Tensor, List[torch.Tensor]],
            images_descript: List[dict],
            batch_size: int = 128,
            device: Optional[Union[str, torch.device]] = None,
            to_np: bool = True,
    ) -> torch.Tensor:
        device = device or get_torch_device("auto")
        if len(queries_embed) == 0:
            raise ValueError("No queries provided")
        if len(images_embed) == 0:
            raise ValueError("No images provided")

        # 首先，为每个图像收集所有相关的文本描述
        image_text_embeds = [[] for _ in range(len(images_embed))]

        for desc in images_descript:
            page = desc['page']
            if 0 <= page < len(images_embed):
                image_text_embeds[page].append(desc['embed'])

        # 准备拼接后的多模态嵌入，保持与images_embed相同的顺序
        multimodal_embeds = []
        for i, image_embed in enumerate(images_embed):
            text_embeds = image_text_embeds[i]

            if text_embeds:
                # 如果存在文本描述，将它们全部拼接到图像嵌入后面
                all_text_embeds = torch.cat(text_embeds, dim=0)
                combined_embed = torch.cat([image_embed, all_text_embeds], dim=0)
            else:
                # 如果没有文本描述，只使用图像嵌入
                combined_embed = image_embed

            multimodal_embeds.append(combined_embed)

        # 转换查询嵌入为张量列表（如果需要）
        if isinstance(queries_embed, torch.Tensor):
            queries_embed = [queries_embed[i] for i in range(queries_embed.shape[0])]

        # 计算相似度得分
        scores_list: List[torch.Tensor] = []
        for i in range(0, len(queries_embed), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                queries_embed[i: i + batch_size], batch_first=True, padding_value=0
            ).to(device)

            for j in range(0, len(multimodal_embeds), batch_size):
                # 获取多模态嵌入批次
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    multimodal_embeds[j: j + batch_size], batch_first=True, padding_value=0
                ).to(device)

                # 计算相似度得分
                scores = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                print(scores.shape)
                scores = scores.max(dim=3)[0].sum(dim=2)
                scores_batch.append(scores)

            # 合并批次结果
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        # 合并所有结果
        scores = torch.cat(scores_list, dim=0)

        # 验证结果形状
        assert scores.shape[0] == len(queries_embed), \
            f"Expected {len(queries_embed)} scores, got {scores.shape[0]}"

        # 转换为numpy数组（如果需要）
        if to_np:
            scores = scores.to(torch.float32).cpu().numpy()

        return scores

    @staticmethod
    def score_mask_vector(
            qs: Union[torch.Tensor, List[torch.Tensor]],
            ps: Union[torch.Tensor, List[torch.Tensor]],
            batch_size: int = 128,
            device: Optional[Union[str, torch.device]] = None,
            to_np: bool = True,
            mask_embed: bool = False,
            images=None,
            queries_ids=None,
            p_length: int = 1024,
            foreground_masks: Optional[List[torch.Tensor]] = None,  # 添加前景掩码参数
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.
            images: List of images for foreground mask detection (if foreground_masks not provided)
            foreground_masks: List of foreground masks for each image (if provided, overrides images)

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        # 直接截断而不是使用掩码
        if mask_embed:
            # 截断passage到p_length
            if isinstance(ps, list):
                ps = [p[:min(p_length, p.size(0))] for p in ps]
            else:
                ps = ps[:, :p_length, :]

            # 截断query，跳过前3个token
            if queries_ids is not None and isinstance(qs, list):
                qs = []
                for i in range(len(queries_ids)):
                    # 找到第一个0的位置
                    zero_indices = (queries_ids[i] == 0).nonzero(as_tuple=True)[0]
                    if zero_indices.numel() > 0:
                        end_idx = zero_indices[0].item()
                    else:
                        end_idx = queries_ids[i].size(0)

                    # 跳过前3个token，截取有效部分
                    start_idx = 3
                    if end_idx > start_idx:
                        qs.append(qs[i][start_idx:end_idx])
                    else:
                        qs.append(torch.zeros(0, qs[i].size(1), device=qs[i].device))

        # 如果没有提供前景掩码但有图像，则检测前景掩码
        if foreground_masks is None and images is not None:
            foreground_masks = []
            for img in images:
                resized_img = img.resize((448, 448))
                mask = detect_foreground_patches(resized_img)
                # 展平为1024维向量，并转换为tensor
                mask_flat = torch.from_numpy(mask.flatten()).bool()
                foreground_masks.append(mask_flat)

        # 如果既没有前景掩码也没有图像，则创建全为True的掩码
        if foreground_masks is None:
            if isinstance(ps, list):
                foreground_masks = [torch.ones(p.shape[0], dtype=torch.bool, device=p.device) for p in ps]
            else:
                foreground_masks = torch.ones(ps.shape[0], ps.shape[1], dtype=torch.bool, device=ps.device)

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []

            # 填充query
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i: i + batch_size], batch_first=True, padding_value=0).to(
                device
            )

            # 创建默认query掩码（所有非零位置为1）
            qs_mask_batch = (torch.sum(qs_batch != 0, dim=-1) != 0).float().to(device)

            for j in range(0, len(ps), batch_size):
                # 填充passage
                if isinstance(ps, list):
                    ps_batch = torch.nn.utils.rnn.pad_sequence(
                        ps[j: j + batch_size], batch_first=True, padding_value=0
                    ).to(device)
                else:
                    ps_batch = ps[j: j + batch_size].to(device)

                # 应用前景掩码：将背景patch的嵌入置为零
                if isinstance(foreground_masks, list):
                    # 获取当前batch的掩码
                    masks_batch = torch.nn.utils.rnn.pad_sequence(
                        foreground_masks[j: j + batch_size], batch_first=True, padding_value=False
                    ).to(device)
                else:
                    masks_batch = foreground_masks[j: j + batch_size].to(device)

                # 应用前景掩码
                masked_ps_batch = ps_batch * masks_batch.unsqueeze(-1)

                # 创建默认passage掩码（所有非零位置为1）
                ps_mask_batch = (torch.sum(masked_ps_batch != 0, dim=-1) != 0).float().to(device)

                # 计算相似度矩阵
                sim_matrix = torch.einsum("bnd,csd->bcns", qs_batch, masked_ps_batch)

                # 应用掩码：将掩码为0的位置设为负无穷，这样在max操作中会被忽略
                # 扩展掩码维度以匹配相似度矩阵
                qs_mask_expanded = qs_mask_batch.unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1)
                ps_mask_expanded = ps_mask_batch.unsqueeze(0).unsqueeze(2)  # (1, C, 1, S)

                # 创建组合掩码
                combined_mask = qs_mask_expanded * ps_mask_expanded

                # 将无效位置的相似度设为负无穷
                sim_matrix_masked = sim_matrix.masked_fill(combined_mask == 0, -float('inf'))

                # 计算最大相似度并求和
                max_sim = sim_matrix_masked.max(dim=3)[0]  # (B, C, N)
                scores = (max_sim * qs_mask_batch.unsqueeze(1)).sum(dim=2)  # (B, C)

                scores_batch.append(scores)

            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        scores = scores.to(torch.float32)
        if to_np:
            scores = scores.to(torch.float32).cpu().numpy()
        return scores


    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass


def detect_foreground_patches(image, patch_size=14, threshold=5):
        """
        检测图像中的前景patch，使用Sobel算子计算梯度

        参数:
            image: PIL Image对象
            patch_size: 每个patch的大小
            threshold: 梯度阈值，高于此值认为是前景

        返回:
            foreground_mask: 32x32的布尔数组，True表示前景patch
        """
        # 转换为灰度图
        gray = np.array(image.convert('L'))

        # 使用Sobel算子计算梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 将图像划分为patch
        h, w = gray.shape
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size

        foreground_mask = np.zeros((num_patches_h, num_patches_w), dtype=bool)

        # 对每个patch计算平均梯度
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size

                patch_gradient = gradient_magnitude[y_start:y_end, x_start:x_end]
                avg_gradient = np.mean(patch_gradient)

                # 如果平均梯度超过阈值，则认为是前景
                if avg_gradient > threshold:
                    foreground_mask[i, j] = True

        return foreground_mask