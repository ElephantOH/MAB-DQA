from typing import ClassVar, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature
from backbones.paligemma import PaliGemmaProcessor
from backbones.colpali.utils.processing_utils import BaseVisualRetrieverProcessor


class ColPaliProcessor(BaseVisualRetrieverProcessor, PaliGemmaProcessor):
    """
    Processor for ColPali.
    """

    visual_prompt_prefix: ClassVar[str] = "<image><bos>Describe the image."
    text_prompt_prefix: ClassVar[str] = "Image description: "
    query_prefix: ClassVar[str] = "Query: "


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPali.
        """
        texts_doc = [self.visual_prompt_prefix] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """

        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.tokenizer.bos_token + self.query_prefix + query
            query += suffix  # add suffix (pad tokens)

            # NOTE: Make input ISO to PaliGemma's processor
            query += "\n"
            # print(query) # <bos>Query: Children<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
            texts_query.append(query)


        batch_query = self.tokenizer(
            texts_query,
            text_pair=None,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        ) # [     2,   5098, 235292,   3118,   3118,      0,      0,      0,      0,     0,      0,      0,      0,      0,      0,    108,      0,      0]
        return batch_query

    def process_texts(
        self,
        texts: List[str],
        max_length: int = 270,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPali.
        """

        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for text in texts:
            text = self.tokenizer.bos_token + self.text_prompt_prefix + text
            text += suffix  # add suffix (pad tokens)

            # NOTE: Make input ISO to PaliGemma's processor
            text += "\n"
            # print(query) # <bos>Query: Children<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
            texts_query.append(text)


        batch_query = self.tokenizer(
            texts_query,
            text_pair=None,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        ) # [     2,   5098, 235292,   3118,   3118,      0,      0,      0,      0,     0,      0,      0,      0,      0,      0,    108,      0,      0]
        return batch_query

    def center_scale_with_white_background(self, batch_image, target_size=(448, 448)):
        """
        中心缩放图像：
        - 若target_size小于当前图片尺寸：缩放图像并填充白色背景至原尺寸
        - 若target_size大于当前图片尺寸：先放大再中心裁剪回原尺寸

        参数:
            batch_image: 包含"pixel_values"的字典，值域为[-1, 1]
            target_size: 目标缩放尺寸，默认为(336, 336)

        返回:
            处理后的batch_image
        """
        # 获取原始尺寸
        original_shape = batch_image.shape
        batch_size, channels, orig_h, orig_w = original_shape
        target_h, target_w = target_size

        # 判断目标尺寸与原始尺寸的关系
        if target_h <= orig_h and target_w <= orig_w:
            # 情况1：目标尺寸小于等于原始尺寸 - 缩放并填充白色背景

            # 缩放图像到目标尺寸
            scaled = F.interpolate(
                batch_image,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

            # 创建全白色背景张量 (白色在[-1,1]范围内表示为1)
            white_background = torch.ones(original_shape, device=batch_image.device)

            # 计算填充位置 (居中放置)
            top = (orig_h - target_h) // 2
            left = (orig_w - target_w) // 2

            # 将缩放后的图像放置到白色背景的中心
            white_background[:, :, top:top + target_h, left:left + target_w] = scaled

            # 更新pixel_values
            batch_image = white_background

        else:
            # 情况2：目标尺寸大于原始尺寸 - 先放大再中心裁剪

            # 放大图像到目标尺寸
            enlarged = F.interpolate(
                batch_image,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

            # 从放大后的图像中心裁剪回原始尺寸
            batch_size, channels, enlarged_h, enlarged_w = enlarged.shape
            top = (enlarged_h - orig_h) // 2
            left = (enlarged_w - orig_w) // 2

            batch_image = enlarged[:, :, top:top + orig_h, left:left + orig_w]

        return batch_image


    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
