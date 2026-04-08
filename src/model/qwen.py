
from src.model.base import BaseVLMClient
from transformers import Qwen2VLForConditionalGeneration, \
                        Qwen2_5_VLForConditionalGeneration, \
                        AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
import os
CURRENT_FILE_PATH = os.path.abspath(__file__)

class QwenConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model_path: str 
    api_key: str = Field(default=None)
    dtype: str = Field(default="bfloat16")
    device_map: str = Field(default="auto")
    use_fast: bool = Field(default=True)
    min_pixels: int = Field(default=256 * 28 * 28)
    max_pixels: int = Field(default=512 * 28 * 28)
    max_new_tokens: int = Field(default=255)


class Qwen2VLClient(BaseVLMClient):
    def __init__(
            self,
            config
        ):
        super().__init__(config)
        try:
            self.config = QwenConfig(**OmegaConf.to_dict(config))
        except:
            self.config = QwenConfig(**OmegaConf.to_container(config, resolve=True))
        self.load_model()

    def load_model(self):
        if self.config.api_key is not None:
            pass
        dtype = getattr(torch, self.config.dtype)
        model_path = f"{self.config.model_path}"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.config.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.config.use_fast,
            # min_pixels=self.config.min_pixels,
            # max_pixels=self.config.max_pixels
        )

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

        print(f"[Info] [{CURRENT_FILE_PATH}] Qwen2 Model Device: {self.model.device}.")
        print(f"[Info] [{CURRENT_FILE_PATH}] Qwen2 Model is on GPU: {next(self.model.parameters()).device}.")

    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})

        message = {
            "role": "user",
            "content": content
        }
        return message

    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    def _resize_image_if_exceeds(self, image, max_size):
        width, height = image.size
        if max(width, height) <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height))

    @torch.no_grad()
    def predict(
        self,
        question,
        texts=None,
        images=None,
        history=None,
        max_new_tokens=None
    ):
        self.clean_up()
        if images is not None:
            images = [self._resize_image_if_exceeds(image, 1000) for image in images]
        messages = self.process_message(question, texts, images, history)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = inputs.to(device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages

    @torch.no_grad()
    def predict_batch(
        self,
        questions,
        texts_list=None,
        images_list=None,
        history_list=None,
    ):
        self.clean_up()

        batch_size = len(questions)

        text_inputs = []
        all_image_inputs = []
        all_video_inputs = []

        for i in range(batch_size):
            texts = texts_list[i] if texts_list else None
            images = images_list[i] if images_list else None
            history = history_list[i] if history_list else None

            messages = self.process_message(questions[i], texts, images, history)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text_inputs.append(text)

            image_inputs, video_inputs = process_vision_info(messages)
            all_image_inputs.append(image_inputs)
            all_video_inputs.append(video_inputs)

        inputs = self.processor(
            text=text_inputs,
            images=all_image_inputs,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self.clean_up()
        return output_texts

    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True


class Qwen25VLClient(Qwen2VLClient):
    def __init__(
            self,
            config,
  
        ):
        super().__init__(config)

    def load_model(self):
        if self.config.api_key is not None:
            pass
        
        dtype = getattr(torch, self.config.dtype)
        model_path = f"{self.config.model_path}"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.config.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            use_fast=self.config.use_fast,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels
        )

        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

        print(f"[Info] [{CURRENT_FILE_PATH}] Qwen2.5 Model Device: {self.model.device}.")
        print(f"[Info] [{CURRENT_FILE_PATH}] Qwen2.5 Model is on GPU: {next(self.model.parameters()).device}.")
     