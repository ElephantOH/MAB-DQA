import os
from typing import Any
import warnings
from omegaconf import DictConfig, OmegaConf
from src.config.config_loader import load_model_config
from src.model.openai import OpenAIClient
from src.model.qwen import Qwen2VLClient, Qwen25VLClient
from src.model.colpali import ColPaliRetriever

CURRENT_FILE_PATH = os.path.abspath(__file__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ModelFactory:

    _MODEL_REGISTRY = {
        "openai/gpt-4o": OpenAIClient,
        "qwen/qwen25vl-3b": Qwen25VLClient,
        "qwen/qwen25vl-7b": Qwen25VLClient,
        "colpali/colpali-1.3": ColPaliRetriever
    }

    @staticmethod
    def _is_supported_primitive(value: Any) -> bool:
        if value is None:
            return True
        primitive_types = (str, int, float, bool, list, dict)
        return isinstance(value, primitive_types)

    @staticmethod
    def create(
        model_type: str,
        config: DictConfig,
        **override_kwargs,
    ) -> Any:
        """
            override_kwargs > model_config(YAML) > configs/base.yaml > config
        """
        assert model_type in ModelFactory._MODEL_REGISTRY, \
            f"[Error] [{CURRENT_FILE_PATH}] Unsupported Model Type: {model_type}, Supported: {list(ModelFactory._MODEL_REGISTRY.keys())}"

        OmegaConf.set_struct(config, False)
        config = OmegaConf.merge(config, load_model_config(model_type=model_type))

        for key, value in override_kwargs.items():
            if ModelFactory._is_supported_primitive(value):
                config[key] = value
            else:
                warnings.warn(
                    f"[Warning] [{CURRENT_FILE_PATH}] Skipped merging config key '{key}': type {type(value).__name__} is not a supported primitive type.",
                    UserWarning
                )
                
        config = OmegaConf.merge(config, override_kwargs)
        OmegaConf.set_struct(config, True)
        OmegaConf.resolve(config)

        try:
            print(f"[Info] [{CURRENT_FILE_PATH}] Using Model: {config.model_path}.")
        except:
            assert config.model_path is not None

        model_cls = ModelFactory._MODEL_REGISTRY[model_type]
        return model_cls(config)
