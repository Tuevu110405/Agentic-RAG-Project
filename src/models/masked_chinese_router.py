import torch
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor
from src.config import config

class NoChineseLogitsProcessor(LogitsProcessor):
    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        # Gán điểm -inf cho các token nằm trong mask
        scores[:, self.mask] = -float("inf")
        return scores

class NoChineseQwenChatModel(LLM):
    model_name: str = config.router_model
    tokenizer: Any = None
    model: Any = None
    chinese_mask: Any = None


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=None,
        )
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return "qwen-2.5 no chinese"

    def _get_chinese_mask(self, vocab_size, device):
        if self.chinese_mask is not None:
            return self.chinese_mask.to(device)

        print("Building Chinese Mask")
        token_ids = torch.arange(vocab_size)
        decoded_tokens = self.tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

        mask = torch.tensor([
            any(0x4E00 <= ord(c) <= 0x9FFF or
                0x3400 <= ord(c) <= 0x4DBF or
                0xF900 <= ord(c) <= 0xFAFF for c in token)
            for token in decoded_tokens
        ], dtype= torch.bool, device = device)

        self.chinese_mask = mask
        print("MASK IS BUILT EFFECTIVELY")
        return self.chinese_mask

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:

        device = self.model.device
        vocab_size = self.model.config.vocab_size
        mask = self._get_chinese_mask(vocab_size, device)
        logits_processor = NoChineseLogitsProcessor(mask)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens = kwargs.get("max_new_tokens",config.router_max_new_tokens),
                temperature = kwargs.get("temperature",config.router_temperature),
                logits_processor =[logits_processor],
                do_sample = True,
                repetition_penalty = 1.1,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id
            )
        output_ids = generated_ids[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response


def get_masked_chinese_router():
    return NoChineseQwenChatModel()






