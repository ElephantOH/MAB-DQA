import os
from typing import ClassVar, Optional, Union

import torch
from safetensors import safe_open
from torch import nn
from backbones.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel, PaliGemmaModel,
)

class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)
        # model = PaliGemmaForConditionalGeneration(config=config)
        model = PaliGemmaModel(config=config)

        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model

        # TODO: Wait for ColPali2 to create a ColPaliConfig to allow specifying the embedding dimension.
        # We could do it now but it would break all the models trying to load the model from the checkpoint.
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)

        self.post_init()

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            *model_args,
            **kwargs,
    ):
        # 首先加载配置
        config = kwargs.pop("config", None)
        # 提取 device_map 和 dtype 参数
        device_map = kwargs.pop("device_map", None)
        dtype = kwargs.pop("dtype", None)
        # 提取其他可能用于设备映射的参数
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_index = kwargs.pop("offload_index", None)

        if config is None:
            config = PaliGemmaConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # 创建模型实例（不加载权重）
        model = cls(config)

        # 应用 dtype 设置
        if dtype is not None:
            model = model.to(dtype)

        # 应用 device_map 设置
        if device_map is not None:
            # 如果 device_map 是 auto, balanced, balanced_low_0 等策略
            if isinstance(device_map, str) and device_map in ["auto", "balanced", "balanced_low_0"]:
                try:
                    from accelerate import infer_auto_device_map, dispatch_model
                except ImportError:
                    raise ImportError(
                        "Using device_map='auto' requires the accelerate library. "
                        "Please install it with `pip install accelerate`."
                    )

                # 推断自动设备映射
                device_map = infer_auto_device_map(
                    model,
                    no_split_module_classes=model._no_split_modules if hasattr(model, '_no_split_modules') else [],
                    dtype=dtype if dtype is not None else next(model.parameters()).dtype,
                    max_memory=max_memory,
                    verbose=True
                )

                # 使用 accelerate 的 dispatch_model
                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_dir=offload_folder,
                    offload_index=offload_index
                )

            # 如果 device_map 是简单的设备字符串
            elif isinstance(device_map, str) and device_map in ["cpu", "cuda", "cuda:0", "cuda:1"]:
                model = model.to(device_map)

            # 如果 device_map 是字典，使用 accelerate 的 dispatch_model
            elif isinstance(device_map, dict):
                try:
                    from accelerate import dispatch_model
                except ImportError:
                    raise ImportError(
                        "Using device_map as a dictionary requires the accelerate library. "
                        "Please install it with `pip install accelerate`."
                    )
                model = dispatch_model(
                    model,
                    device_map=device_map,
                    offload_dir=offload_folder,
                    offload_index=offload_index
                )
            else:
                raise ValueError(
                    f"device_map must be a string ('auto', 'balanced', 'balanced_low_0', 'cpu', 'cuda', etc.) or dict, got {type(device_map)}")

        # 获取权重文件路径
        if os.path.isdir(pretrained_model_name_or_path):
            # 本地路径
            model_dir = pretrained_model_name_or_path
        else:
            # 可能是模型名称，需要下载（这里简化处理）
            # 实际实现中可能需要使用 cached_file 或类似函数
            model_dir = pretrained_model_name_or_path
            assert False, f"pretrained_model_name_or_path={pretrained_model_name_or_path} does not exist"

        # 查找所有的 safetensors 文件
        safetensors_files = []
        if os.path.exists(os.path.join(model_dir, "model.safetensors")):
            safetensors_files.append(os.path.join(model_dir, "model.safetensors"))
        elif os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
            # 处理分片权重
            import json
            with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
                index = json.load(f)
                # 使用集合去重
                unique_weight_files = set(index["weight_map"].values())
                for weight_file in unique_weight_files:
                    safetensors_files.append(os.path.join(model_dir, weight_file))
        else:
            # 尝试查找其他可能的文件
            for f in os.listdir(model_dir):
                if f.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(model_dir, f))

        if not safetensors_files:
            # 如果找不到 safetensors 文件，尝试查找 pytorch_model.bin 文件
            pytorch_bin_files = []
            if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
                pytorch_bin_files.append(os.path.join(model_dir, "pytorch_model.bin"))
            elif os.path.exists(os.path.join(model_dir, "pytorch_model.bin.index.json")):
                import json
                with open(os.path.join(model_dir, "pytorch_model.bin.index.json"), "r") as f:
                    index = json.load(f)
                    unique_weight_files = set(index["weight_map"].values())
                    for weight_file in unique_weight_files:
                        pytorch_bin_files.append(os.path.join(model_dir, weight_file))

            if pytorch_bin_files:
                # 加载 pytorch_model.bin 文件
                state_dict = {}
                for file_path in pytorch_bin_files:
                    state_dict.update(torch.load(file_path, map_location="cpu"))
            else:
                raise ValueError(f"No safetensors or pytorch_model.bin files found in {model_dir}")
        else:
            # 加载并合并所有权重
            state_dict = {}
            for file_path in safetensors_files:
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

        # 重映射权重键
        new_state_dict = {}
        replace_flag = False

        for key, value in state_dict.items():
            if key.startswith('model.language_model.model.'):
                replace_flag = True
                # 替换前缀
                new_key = key.replace('model.language_model.model.', 'model.language_model.')
                new_state_dict[new_key] = value
            else:
                # 保持其他权重不变
                new_state_dict[key] = value

        if replace_flag:
            print("[Info] Replacing model parameters.")

        # 加载处理后的权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(f"[Error] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[Error] Unexpected keys: {unexpected_keys}")

        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return proj

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        return self.model.vision_tower.config.patch_size
