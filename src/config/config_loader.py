import os
from omegaconf import OmegaConf, DictConfig

CURRENT_FILE_PATH = os.path.abspath(__file__)
BASE_CONFIG_NAME = "base.yaml"

def _check_file_exists(file_path: str) -> None:
    """
    Check if the file exists, raise an assertion error in the specified format if it does not exist
    :param file_path: File path to be checked
    """
    assert os.path.exists(file_path), \
        f'[Error] [{CURRENT_FILE_PATH}] "Config File Not Found At Path: {file_path}."'

def _load_recursive_base_configs(config_root: str, target_sub_dir: str) -> DictConfig:
    """
    Traverse directories recursively upwards and load all existing base.yaml configurations (supports multi-level directories)
    :param config_root: Root directory of configurations (e.g., configs)
    :param target_sub_dir: Target subdirectory (e.g., model/qwen/qwen2.5vl)
    :return: Merged all base configurations (return empty config if none)
    """
    base_configs = []
    current_dir = os.path.join(config_root, target_sub_dir)
    
    while current_dir.startswith(config_root) and os.path.isdir(current_dir):
        base_file = os.path.join(current_dir, BASE_CONFIG_NAME)
        if os.path.exists(base_file):
            base_configs.append(OmegaConf.load(base_file))
        current_dir = os.path.dirname(current_dir)

    return OmegaConf.merge(*base_configs) if base_configs else OmegaConf.create()

def _load_and_merge_config(
    config_root: str,
    target_sub_path: str,
) -> DictConfig:
    """
    General configuration loading and merging logic (core reusable function, modify the calling layer only for new parameters)
    :param config_root: Root directory of configurations
    :param target_sub_path: Subpath of the target configuration file (supports /)
    :return: Final merged configuration
    """
    target_file = f"{os.path.join(config_root, target_sub_path)}.yaml"
    _check_file_exists(target_file)
    target_config = OmegaConf.load(target_file)
    target_dir = os.path.dirname(target_sub_path)
    base_config = _load_recursive_base_configs(config_root, target_dir)
    return OmegaConf.merge(base_config, target_config)

#%%
def load_model_config(
    config_name: str = "configs",
    directory_key: str = "model",
    model_type: str = "base",
) -> DictConfig:
    """
    Load model configuration (supports model_type containing /, e.g., qwen/qwen2.5vl)
    """
    target_sub_path = os.path.join(directory_key, model_type)
    return _load_and_merge_config(config_name, target_sub_path)

def load_dataset_config(
    config_name: str = "configs",
    directory_key: str = "dataset",
    mission_key: str = "base",
    dataset_type: str = "base",
) -> DictConfig:
    """
    Load dataset configuration (supports dataset_type containing / and multi-level base.yaml)
    """
    target_sub_path = os.path.join(directory_key, mission_key, dataset_type)
    return _load_and_merge_config(config_name, target_sub_path)
