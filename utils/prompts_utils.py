from utils.models_utils import ModelSpec

def build_messages(prompt: str, spec: ModelSpec, want_thinking: bool = False) -> list:
    """
    Wrap the annotation prompt in the chat structure the model expects.
    For models using Nemotron-style reasoning control, prepend a system message
    with "/think" or "/no_think" per the model's documented convention.
    Otherwise, no custom system prompt: the instruction lives entirely in the user
    turn, and the model's own default system prompt (if any) is left in place by
    its chat template.
    """
    messages = []

    if spec.thinking_control == "system_flag":
        flag = "/think" if want_thinking else "/no_think"
        messages.append({"role": "system", "content": flag})

    if spec.is_vlm:
        content = [{"type": "text", "text": prompt}]
    else:
        content = prompt
    messages.append({"role": "user", "content": content})

    return messages


def apply_chat_template_safe(tok, messages, spec: ModelSpec, want_thinking: bool = False):
    """Apply the chat template, using the model-appropriate thinking-control mechanism."""
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if spec.thinking_control == "template_kwarg":
        kwargs["enable_thinking"] = want_thinking
    try:
        return tok.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tok.apply_chat_template(messages, **kwargs)