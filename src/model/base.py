import torch
import os
import abc
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseVLMModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    def clean_up(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseVLMRetriever(BaseVLMModel):
    @abstractmethod
    def encode_images(self, images: List, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        pass


class BaseVLMClient(BaseVLMModel):

    @abc.abstractmethod
    def predict(self, question, texts=None, images=None, history=None, **kwargs):
        pass
        
    def process_message(self, question, texts, images, history):
        if history is not None:
            assert(self.is_valid_history(history))
            messages = history
        else:
            messages = []
        
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(self.create_image_message(images, question))
        
        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))
        
        return messages
    
    def is_valid_history(self, history):
        return True
    
    def clean_up(self):
        torch.cuda.empty_cache()