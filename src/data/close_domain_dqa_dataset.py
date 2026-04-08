
import os
CURRENT_FILE_PATH = os.path.abspath(__file__)

import re
import json
import numpy as np
import torch

import pymupdf
from PIL import Image
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.data.base import DQADataset
from collections import defaultdict

class CloseDomainDQADatasetConfig(BaseModel):
    
    model_config = {"arbitrary_types_allowed": True}
    dataset_name: str
    dataset_path: str
    sample_origin_path: str
    sample_dump_path: str
    document_path: str
    result_path: str = Field(default="results")
    work_path: str = Field(default="work")

    question_key: str = Field(default="question")
    retrieval_key: str = Field(default="retrieval")
    answer_key: str = Field(default="answer")
    
    max_character_per_page: int = Field(default=10000)
    max_page: int = Field(default=5000)

#%%
@dataclass
class Content:
    image: Image
    text: str

class CloseDomainDQADataset(DQADataset):
    def __init__(self, config):

        super().__init__(config)
        try:
            self.config = CloseDomainDQADatasetConfig(**OmegaConf.to_dict(config))
        except:
            self.config = CloseDomainDQADatasetConfig(**OmegaConf.to_container(config, resolve=True))

        self.dataset_name = config.dataset_name
        self.dataset_path = config.dataset_path
        self.sample_origin_path = config.sample_origin_path
        self.sample_dump_path = config.sample_dump_path
        self.document_path = config.document_path
        self.result_path = config.result_path
        self.work_path = config.work_path

        self.question_key = config.question_key
        self.retrieval_key = config.retrieval_key
        self.answer_key = config.answer_key

        self.CLIP_IMAGE_FILE = lambda doc_name, index: f"{self.work_path}/image/{doc_name}/clip_{index}.png"
        self.IMAGE_FILE = lambda doc_name, index: f"{self.work_path}/image/{doc_name}/{index}.png"
        self.TEXT_FILE = lambda doc_name, index: f"{self.work_path}/text/{doc_name}/{index}.txt"
        self.CLIP_IMAGE_EMBED = lambda doc_name: f"{self.work_path}/embed/{doc_name}/clip_image_embed.pt"
        self.IMAGE_EMBED = lambda doc_name: f"{self.work_path}/embed/{doc_name}/image_embed.pt"
        self.EXTRACT_DOCUMENT_ID = lambda sample: re.sub("\\.pdf$", "", sample["doc_id"]).split("/")[-1]

    def load_samples_file(self, sample_name=None, suffix=None):

        if suffix is not None:
            path = self.sample_dump_path
        else:
            path = self.sample_origin_path
        if sample_name is not None or suffix is not None:
            dirname = os.path.dirname(path)
            name, ext = os.path.splitext(os.path.basename(path))
            if sample_name is not None:
                name = sample_name
            target_path = os.path.join(dirname, f"{name}{suffix}{ext}") if suffix else None
            default_path = os.path.join(dirname, f"{name}{ext}")
            
            if target_path and os.path.exists(target_path):
                path = target_path
            elif os.path.exists(default_path):
                path = default_path
            else:
                print(f"[Warning] [{CURRENT_FILE_PATH}] {target_path or default_path} no Exists.")

        print(f"[Info] [{CURRENT_FILE_PATH}] Use Dataset Path: {path}.")
        assert os.path.exists(path)
        with open(path, 'r') as f:
            samples = json.load(f)
        return samples

    def dump_samples_file(self, samples, sample_name=None, suffix=None):

        if suffix is not None:
            path = self.sample_dump_path
        else:
            path = self.sample_origin_path
        
        dirname = os.path.dirname(path)
        base_name, ext = os.path.splitext(os.path.basename(path))

        if sample_name:
            base_name = sample_name

        if suffix:
            final_name = f"{base_name}{suffix}{ext}"
        else:
            final_name = f"{base_name}{ext}"

        path = os.path.join(dirname, final_name)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(samples, f, indent=4)
        return path

    def group_samples_by_doc_id(self, samples):

        grouped_samples = defaultdict(list)
        for sample in samples:
            grouped_samples[sample["doc_id"]].append(sample)
        return grouped_samples

    def get_retrieval_results(self, sample, top_k=10):

        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        question: str = sample[self.question_key]
        texts = []
        images = []
        text_pages=[]
        image_pages=[]

        if self.retrieval_key in sample:
            for page in sample[self.retrieval_key][:top_k]:
                text_file = self.TEXT_FILE(doc_name, page)
                with open(text_file, 'r') as file:
                    text = file.read()
                text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')[:self.config.max_character_per_page]
                texts.append(text)
                text_pages.append(page)

        if self.retrieval_key in sample:
            for page in sample[self.retrieval_key][:top_k]:
                image_file = self.IMAGE_FILE(doc_name, page)
                image = Image.open(image_file)
                images.append(image)
                image_pages.append(page)

        return question, texts, images, text_pages, image_pages

    def load_image_embeds(self, sample, is_clip=True):

        doc_name = self.EXTRACT_DOCUMENT_ID(sample)

        if is_clip:
            image_embeds_path = self.CLIP_IMAGE_EMBED(doc_name)
        else:
            image_embeds_path = self.IMAGE_EMBED(doc_name)

        if not os.path.exists(image_embeds_path):
            print(f"[Warning] [{CURRENT_FILE_PATH}] Embedding File not Found: {image_embeds_path}")
            return None

        try:
            embedding = torch.load(image_embeds_path, map_location='cpu')
            if embedding is None or (torch.is_tensor(embedding) and embedding.numel() == 0):
                raise ValueError(f"[Error] [{CURRENT_FILE_PATH}] Embedding is Empty or Invalid.")
            return embedding
        except Exception as e:
            print(f"[Error] [{CURRENT_FILE_PATH}] Failed to Load Embedding {image_embeds_path}: {e}.")
            try:
                os.remove(image_embeds_path)
                print(f"[Info] [{CURRENT_FILE_PATH}] Removed Corrupted File: {image_embeds_path}.")
            except:
                pass
            return None

    def save_image_embeds(self, sample, image_embeds, is_clip=True):

        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        if is_clip:
            image_embeds_path = self.CLIP_IMAGE_EMBED(doc_name)
        else:
            image_embeds_path = self.IMAGE_EMBED(doc_name)
        os.makedirs(os.path.dirname(image_embeds_path), exist_ok=True)
        torch.save(image_embeds, image_embeds_path)

    def extract_document_contents(
            self,
            sample,
            load_contents=True,
            save_contents=True,
            clip_border=False,
            remove_inner=False,
            dpi=180
        ):
        contents = []
        doc_name = self.EXTRACT_DOCUMENT_ID(sample)

        with pymupdf.open(os.path.join(self.document_path, sample["doc_id"])) as pdf:
            for index, page in enumerate(pdf[:self.config.max_page]):
                image = page.get_pixmap(dpi=dpi)
                if clip_border:
                    image = self.remove_image_border(image, remove_inner=remove_inner)
                text = page.get_text("text")

                if save_contents is True:
                    if clip_border:
                        image_file = self.CLIP_IMAGE_FILE(doc_name, index)
                    else:
                        image_file = self.IMAGE_FILE(doc_name, index)
                    text_file = self.TEXT_FILE(doc_name, index)
                    os.makedirs(os.path.dirname(image_file), exist_ok=True)
                    os.makedirs(os.path.dirname(text_file), exist_ok=True)
                    image.save(image_file)
                    with open(text_file, 'w') as f:
                        f.write(text)

                if hasattr(image, 'samples') and hasattr(image, 'width') and hasattr(image, 'height'):
                    mode = "RGB"
                    if image.alpha:
                        mode = "RGBA"
                    size = (image.width, image.height)
                    image = Image.frombytes(mode, size, image.samples)
                contents.append(Content(image=image, text=text))
        return contents
    
    def extract_page_contents(
            self,
            sample,
            page,
            load_contents=True,
            save_contents=True,
            clip_border=False,
            remove_inner=False,
            dpi=180
        ):

        doc_name = self.EXTRACT_DOCUMENT_ID(sample)
        if clip_border:
            image_path = self.CLIP_IMAGE_FILE(doc_name, page)
        else:
            image_path = self.IMAGE_FILE(doc_name, page)
        text_path = self.TEXT_FILE(doc_name, page)

        if load_contents and os.path.exists(image_path) and os.path.exists(text_path):
            image = Image.open(image_path)
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            pdf_path = os.path.join(self.document_path, sample["doc_id"])
            with pymupdf.open(pdf_path) as pdf:
                pdf_page = pdf[page]
                pixmap = pdf_page.get_pixmap(dpi=dpi)
                if clip_border:
                    pixmap = self.remove_image_border(pixmap, remove_inner=remove_inner)
                text = pdf_page.get_text("text")

                if save_contents:
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    os.makedirs(os.path.dirname(text_path), exist_ok=True)
                    pixmap.save(image_path)
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                if hasattr(pixmap, 'samples') and hasattr(pixmap, 'width') and hasattr(pixmap, 'height'):
                    mode = "RGB"
                    if pixmap.alpha:
                        mode = "RGBA"
                    size = (pixmap.width, pixmap.height)
                    image = Image.frombytes(mode, size, pixmap.samples)

        contents = [Content(image=image, text=text)]
        return contents

    def get_pdf_length(self, sample):
        length = 0
        with pymupdf.open(os.path.join(self.document_path, sample["doc_id"])) as pdf:
            length = len(pdf)
        return length

    def remove_image_border(self, image, tolerance=10, remove_inner=True, margin=1):
        if hasattr(image, 'samples') and hasattr(image, 'width') and hasattr(image, 'height'):
            image_mode = "RGB"
            if image.alpha:
                image_mode = "RGBA"
            pil_image = Image.frombytes(image_mode, [image.width, image.height], image.samples)
        else:
            pil_image = image

        image_np_array = np.array(pil_image)
        image_height, image_width, color_channels = image_np_array.shape

        corner_pixels = [
            image_np_array[0, 0],
            image_np_array[0, image_width - 1],
            image_np_array[image_height - 1, 0],
            image_np_array[image_height - 1, image_width - 1]
        ]
        background_color = np.mean(corner_pixels, axis=0)
        color_difference = np.abs(image_np_array - background_color)
        is_background_pixel = np.all(color_difference <= tolerance, axis=2)

        if remove_inner:
            if np.all(is_background_pixel):
                return image

            non_background_rows = np.where(~np.all(is_background_pixel, axis=1))[0]
            non_background_cols = np.where(~np.all(is_background_pixel, axis=0))[0]
            if len(non_background_rows) == 0 or len(non_background_cols) == 0:
                return pil_image.crop((0, 0, 1, 1))

            cropped_image_array = image_np_array[non_background_rows, :, :]
            cropped_image_array = cropped_image_array[:, non_background_cols, :]
            cropped_pil_image = Image.fromarray(cropped_image_array)

            if margin > 0:
                padded_width = cropped_image_array.shape[1] + 2 * margin
                padded_height = cropped_image_array.shape[0] + 2 * margin
                padded_pil_image = Image.new(pil_image.mode, (padded_width, padded_height), tuple(background_color.astype(int)))
                padded_pil_image.paste(cropped_pil_image, (margin, margin))
                return padded_pil_image
            return cropped_pil_image

        is_full_background_row = np.all(is_background_pixel, axis=1)
        top_bound = np.argmax(~is_full_background_row)
        bottom_bound = image_height - 1 - np.argmax(~is_full_background_row[::-1])

        is_full_background_col = np.all(is_background_pixel, axis=0)
        left_bound = np.argmax(~is_full_background_col)
        right_bound = image_width - 1 - np.argmax(~is_full_background_col[::-1])

        if top_bound >= bottom_bound or left_bound >= right_bound:
            return pil_image.crop((0, 0, 1, 1))

        top_bound = max(0, top_bound - margin)
        bottom_bound = min(image_height, bottom_bound + margin + 1)
        left_bound = max(0, left_bound - margin)
        right_bound = min(image_width, right_bound + margin + 1)

        return pil_image.crop((left_bound, top_bound, right_bound, bottom_bound))