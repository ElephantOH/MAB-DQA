from typing import List
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pydantic import BaseModel, Field
from backbones.colpali.models import ColPali, ColPaliProcessor
from src.model.base import BaseVLMRetriever


class ColPaliConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    model_path: str
    
    api_key: str = Field(default=None)
    dtype: str = Field(default="bfloat16")
    device_map: str = Field(default="auto")
    use_fast: bool = Field(default=True)

class ColPaliRetriever(BaseVLMRetriever):
    def __init__(self, config):
        super().__init__(config)
        try:
            self.config = ColPaliConfig(**OmegaConf.to_dict(config))
        except:
            self.config = ColPaliConfig(**OmegaConf.to_container(config, resolve=True))
        self.load_model()

    def load_model(self):
        if self.config.api_key is not None:
            pass
        dtype = getattr(torch, self.config.dtype)
        model_path = f"{self.config.model_path}"
        
        self.model = ColPali.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.config.device_map
        ).eval()
        
        self.processor = ColPaliProcessor.from_pretrained(
            model_path,
            use_fast=self.config.use_fast
        )

    def encode_images(
            self,
            images: List,
            batch_size: int=1,
            **kwargs    
        ) -> torch.Tensor:
        with torch.no_grad():
            dataloader = DataLoader(
                images,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x).to(self.model.device),
            )

            image_embeds = []
            for batch_image in dataloader:
                batch_image = {k: v.to(self.model.device) for k, v in batch_image.items()}
                batch_image_embed = self.model(**batch_image)
                image_embeds.extend(batch_image_embed)

            image_embeds = torch.stack(image_embeds, axis=0)
            return image_embeds

    def encode_queries(
            self,
            queries: List[str]
        ) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor.process_queries(queries).to(self.model.device)
            return self.model(**inputs)
