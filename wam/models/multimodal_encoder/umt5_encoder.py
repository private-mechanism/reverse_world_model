import os

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def _is_hf_deepspeed_zero3() -> bool:
    """ZeRO-3 下 HF 禁止 ``low_cpu_mem_usage`` / ``device_map``，需常规加载再 ``.to(device)``。"""
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled

        return bool(is_deepspeed_zero3_enabled())
    except ImportError:
        return False


def _build_t5_from_pretrained_kwargs(
    device: torch.device,
    torch_dtype: torch.dtype,
    *,
    use_offload_folder=None,
    t5_model_kwargs=None,
):
    if t5_model_kwargs is not None:
        return dict(t5_model_kwargs)

    if _is_hf_deepspeed_zero3():
        # 与 train_world + accelerate launch + zero3.json 同跑时走此分支
        return {"torch_dtype": torch_dtype}

    kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }
    if use_offload_folder is not None:
        kwargs["offload_folder"] = use_offload_folder
        kwargs["device_map"] = {
            "shared": device,
            "encoder.embed_tokens": device,
            "encoder.block.0": device,
            "encoder.block.1": device,
            "encoder.block.2": device,
            "encoder.block.3": device,
            "encoder.block.4": device,
            "encoder.block.5": device,
            "encoder.block.6": device,
            "encoder.block.7": device,
            "encoder.block.8": device,
            "encoder.block.9": device,
            "encoder.block.10": device,
            "encoder.block.11": device,
            "encoder.block.12": "disk",
            "encoder.block.13": "disk",
            "encoder.block.14": "disk",
            "encoder.block.15": "disk",
            "encoder.block.16": "disk",
            "encoder.block.17": "disk",
            "encoder.block.18": "disk",
            "encoder.block.19": "disk",
            "encoder.block.20": "disk",
            "encoder.block.21": "disk",
            "encoder.block.22": "disk",
            "encoder.block.23": "disk",
            "encoder.final_layer_norm": "disk",
            "encoder.dropout": "disk",
        }
    else:
        kwargs["device_map"] = {
            "shared": device,
            "encoder": device,
        }
    return kwargs


class umT5Embedder:
    # available_models = ["google/t5-v1_1-xxl"]

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=512,
        local_files_only=False,
    ):
        # from_pretrained="google/t5-v1_1-xxl" # zijian
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        load_kwargs = _build_t5_from_pretrained_kwargs(
            self.device,
            self.torch_dtype,
            use_offload_folder=use_offload_folder,
            t5_model_kwargs=t5_model_kwargs,
        )
        self._loaded_with_device_map = "device_map" in load_kwargs

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        # assert from_pretrained in self.available_models
        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            subfolder="tokenizer",
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=False,  # 核心修改：禁用快速分词器
            trust_remote_code=True  # 可选：如果是自定义 T5 模型
        )
        self.model = UMT5EncoderModel.from_pretrained(
            from_pretrained,
            subfolder="text_encoder",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            # use_safetensors=False,
            **load_kwargs,
        ).eval()
        if not self._loaded_with_device_map:
            self.model.to(self.device, dtype=self.torch_dtype)
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        text_encoder_embs = [u[:v] for u, v in zip(text_encoder_embs, seq_lens)]
        text_encoder_embs = torch.stack(
            [torch.cat([u, u.new_zeros(self.model_max_length - u.size(0), u.size(1))]) for u in text_encoder_embs], dim=0
        )
        return text_encoder_embs, attention_mask


if __name__ == "__main__":
    encoder = umT5Embedder(
        from_pretrained="/mnt/dataset/projs/pretrained_models/WoW-1-Wan-1.3B-2M-Diffusers", 
        device='cuda:1')
    text_embeddings = encoder.get_text_embeddings(["Hello, world!"])
    print(f" text_embeddings.shape          = ", text_embeddings[0].shape) # torch.Size([1, 512, 4096])
    input_ids = torch.tensor([[78637,292,312,48694,13706,80959,301,289,9934,280,1753,7868,1,0,0,0,0,0,0,0,0]])
    text_embeddings = encoder.model(input_ids=input_ids.to("cuda:1"), attention_mask=None)["last_hidden_state"].detach()
    print(f" text_embeddings.shape          = ", text_embeddings[0].shape) # torch.Size([21, 4096])
