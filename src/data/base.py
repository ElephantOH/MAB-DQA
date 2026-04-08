

from PIL import Image
from dataclasses import dataclass
from abc import ABC, abstractmethod

#%%
@dataclass
class Content:
    image: Image
    text: str

class DQADataset(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def extract_document_contents(self, sample, load_contents=True, save_contents=True, **kwargs):
        pass

    @abstractmethod
    def extract_page_contents(self, sample, page, load_contents=True, save_contents=True, **kwargs):
        pass

    @abstractmethod
    def load_samples_file(self, sample_name=None, suffix=None, **kwargs):
        pass

    @abstractmethod
    def dump_samples_file(self, samples, sample_name=None, suffix=None, **kwargs):
        pass

    @abstractmethod
    def get_retrieval_results(self, sample, top_k=10, **kwargs):
        pass
