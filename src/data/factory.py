import os
from typing import Any
from omegaconf import DictConfig, OmegaConf
from src.config.config_loader import load_dataset_config
from src.data.close_domain_dqa_dataset import CloseDomainDQADataset
from src.data.open_domain_dqa_dataset import OpenDomainDQADataset

CURRENT_FILE_PATH = os.path.abspath(__file__)

class DatasetFactory:

    _MISSION_REGISTRY = {
        "close_domain_dqa": CloseDomainDQADataset,
        "open_domain_dqa": OpenDomainDQADataset,
    }

    _Dataset_REGISTRY = {
        "mmlb": None,
        "ldu": None,
        "ptab": None,
        "feta": None,
    }

    @staticmethod
    def create(
        mission_key: str,
        dataset_type: str,
        config: DictConfig,
        **override_kwargs,
    ) -> Any:
        """
            override_kwargs > dataset_config(YAML) > configs/base.yaml > config
        """
        assert mission_key in DatasetFactory._MISSION_REGISTRY, \
            f"[Error] [{CURRENT_FILE_PATH}] Unsupported Mission Key: {mission_key}, Supported: {list(DatasetFactory._MISSION_REGISTRY.keys())}"
        
        assert dataset_type in DatasetFactory._Dataset_REGISTRY, \
            f"[Error] [{CURRENT_FILE_PATH}] Unsupported Dataset Type: {dataset_type}, Supported: {list(DatasetFactory._Dataset_REGISTRY.keys())}"

        OmegaConf.set_struct(config, False)
        config = OmegaConf.merge(config, load_dataset_config(mission_key=mission_key, dataset_type=dataset_type))
        config = OmegaConf.merge(config, override_kwargs)
        OmegaConf.set_struct(config, True)
        OmegaConf.resolve(config)

        try:
            print(f"[Info] [{CURRENT_FILE_PATH}] Using Dataset: {config.dataset_name}.")
        except:
            assert config.dataset_name is not None

        model_cls = DatasetFactory._MISSION_REGISTRY[mission_key]
        return model_cls(config)
