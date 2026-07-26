import os
import torch
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

@dataclass
class ModelSpec:
    name: str
    model_class: type
    proc_class: type
    is_vlm: bool
    supports_system_role: bool
    # How to suppress the model's reasoning/thinking mode, if it has one:
    #   None            -> model has no thinking mode, do nothing
    #   "system_flag"   -> inject a system message containing "/think" or "/no_think" (Nemotron-style)
    #   "template_kwarg" -> pass enable_thinking=True/False to apply_chat_template (Qwen-style)
    thinking_control: str | None = None
    load_kwargs: dict = field(default_factory=dict)
    proc_kwargs: dict = field(default_factory=dict)
    batch_size_divide: int = 1


# Custom flags that live in proc_kwargs but aren't real HF from_pretrained kwargs.
# They're popped out before instantiation and handled manually in load_model().
CUSTOM_PROC_FLAGS = {"fix_mistral_regex"}

# Best models selected zero-shot from the Evalita-LLM leaderboard (decreasing order by overall performance).
REGISTRY = {
    "gemma3": ModelSpec(
        "google/gemma-3-27b-it",
        Gemma3ForConditionalGeneration, AutoProcessor,
        is_vlm=True, supports_system_role=True,  # default system prompt: "You are a helpful assistant."
        batch_size_divide=1,
    ),
    "llama-3.3-70b": ModelSpec(
        "meta-llama/Llama-3.3-70B-Instruct",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
        batch_size_divide=2,
    ),
    "qwen2.5-72b": ModelSpec(
        "Qwen/Qwen2.5-72B-Instruct",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
        batch_size_divide=2,
    ),
    "mistral-24b": ModelSpec(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
        proc_kwargs={"fix_mistral_regex": True},
        batch_size_divide=1,
    ),

    # For small trials
    "llama-3.1-8b": ModelSpec(
        "meta-llama/Llama-3.1-8B-Instruct",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
        batch_size_divide=1,
    )
}



def load_model(spec: ModelSpec):
    """
    Instantiate model + tokenizer/processor from a ModelSpec.
    Uses the explicit class in the spec; applies bf16 + auto sharding globally.
    Sets left padding for batched gen. Applies any model-specific quirk fixes.
    """
    HF_TOKEN = os.getenv("HF_TOKEN")

    hf_proc_kwargs = {k: v for k, v in spec.proc_kwargs.items() if k not in CUSTOM_PROC_FLAGS}
    proc = spec.proc_class.from_pretrained(spec.name, token=HF_TOKEN, **hf_proc_kwargs)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "balanced",
        **spec.load_kwargs,  # spec wins on conflict
    }
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN, **load_kwargs)
    model.eval()

    # For a VLM the real tokenizer is proc.tokenizer; for a text model proc IS it.
    tok = proc.tokenizer if spec.is_vlm else proc

    # Known Mistral fast-tokenizer regex issue on some batched inputs: fall back to slow tokenizer.
    if spec.proc_kwargs.get("fix_mistral_regex") and getattr(tok, "is_fast", False):
        try:
            slow_tok = AutoTokenizer.from_pretrained(spec.name, token=HF_TOKEN, use_fast=False)
            if spec.is_vlm:
                proc.tokenizer = slow_tok
            else:
                proc = slow_tok
            tok = slow_tok
        except Exception as e:
            print(f"Warning: could not apply mistral regex fix, continuing with fast tokenizer: {e}")

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc