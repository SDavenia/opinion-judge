from utils.models_utils import ModelSpec

def build_messages(prompt: str, spec: ModelSpec, want_thinking: bool = False) -> list:
    """
    Wrap the annotation prompt in the chat structure the model expects.
    Also disables thinking for thinking models.
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

GENERATION_PROMPTS = {
    "base": """You are a precise text generation engine for an NLP evaluation dataset.
Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}
Task:
{stance_instruction}

Constraints:
1. The output must be exactly 2-3 sentences long.
2. Write in a natural, first-person or third-person argumentative tone (as if written by a human expressing a genuine opinion).
3. Do not explicitly mention the words "Valence", "Core Rationale", or quote the instructions. Integrate the Core Rationale seamlessly into the stance.
Output only the opinion text.""",

    "human_simulator": """You are a person that is asked to write an opinion on a moral situation. You are given a situation, a moral principle, and a target relationship to the situation (Supports, Opposes, or Either). 
You are also given a core rationale that explains why the moral principle is relevant to the situation. 
Your task is to write an opinion that reflects the target relationship to the situation, using the moral principle and the core rationale.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}
Task:
{stance_instruction}

Constraints:
1. The output must be exactly 2-3 sentences long.
2. Write in a natura, first person argumentative tone (as if written by a human expressing a genuine opinion).
3. Do not explicitly mention the words "Valence", "Core Rationale", or quote the instructions. Integrate the Core Rationale seamlessly into the stance.
Output only the opinion text.""",

    "reflective_person": """Imagine you are sharing your honest thoughts with someone after reading about a situation.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}

Task:
{stance_instruction}

Write a short opinion that genuinely sounds like something a person would say. Let the moral principle shape your reasoning naturally rather than naming it explicitly. The opinion should feel thoughtful and personal, not like an explanation for a dataset.

Constraints:
1. Exactly 2-3 sentences.
2. Use a natural first-person argumentative voice.
3. Integrate the rationale implicitly.
4. Avoid sounding formal, academic, or overly balanced unless the required Target Relationship is "Either".
Output only the opinion.""",


"internal_monologue": """You have just read the following situation and are reacting to it in your own mind.

Situation: {situation}
Moral Principle: {value}
Target Relationship to Situation: {valence}
Core Rationale: {explanation}

Task:
{stance_instruction}

Write what you genuinely think. Your reasoning should emerge naturally from your values instead of sounding like a structured argument.

Constraints:
1. Exactly 2-3 sentences.
2. Write as an authentic internal reflection.
3. Don't mention the moral principle directly unless it would naturally come up.
4. Make the opinion feel emotionally believable rather than analytical.
Output only the opinion."""

}

SCORING_PROMPTS = {
    "base": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only a score from 1 to 4.""",

    "base-nofluency": """You are an impartial evaluator
Assess how well the second opinion (Opinion B) aligns with the first (Opinion A) in the following situation.
Rate how aligned they are in terms of viewpoint, on a scale of [1-4].
Do not consider writing quality, length, or fluency, only the similarity of position and underlying reasoning.

Situation: {situation}
Opinion A: {first_opinion}
Opinion B: {second_opinion}

Return only a score from 1 to 4.""",

    "base-with-scale": """Rate how well Opinion B reflects Opinion A on a 1-4 scale:
4 = same overall position, grounded in essentially the same underlying value/reason
3 = same overall position, but grounded in a different underlying value/reason
2 = one opinion takes a clear position, the other is neutral or mixed
1 = the opinions take opposing positions

Situation: {situation}
Opinion A: {first_opinion}
Opinion B: {second_opinion}

Return only the number.""",

    "0_1": {
        "base": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
You have to assign a score in the range [0,1] following this criteria:
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — No alignment: The opinions have fundamentally incompatible stances. 

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only the score.""",

        "lex_var": """You are a helpful and meticulous evaluator.
You have to assess how well the second opinion aligns with the first in a given situation.
You have to assign a score in the range [0,1] following the criteria:
-1.0 — The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — The opinions have fundamentally incompatible stances. 

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Answer only with the score.""",

        "reverse_order": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
You have to assign a score in the range [0,1] following this criteria:
-0.0 — No alignment: The opinions have fundamentally incompatible stances. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Return only the score.""",

        "lex_var_reverse_order": """You are a helpful and meticulous evaluator.
You have to assess how well the second opinion aligns with the first in a given situation.
You have to assign a score in the range [0,1] following the criteria:
-0.0 — The opinions have fundamentally incompatible stances. 
-[0.1, 0.3] — The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-[0.4, 0.6] — The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.7, 0.9] — The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-1.0 — The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.

Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

Answer only with the score.""",

        "criteria_after": """You are an impartial evaluator.
Assess how well the second opinion aligns with the first in the following situation.
Situation: {situation}
First opinion: {first_opinion}
Second opinion: {second_opinion}

You have to assign a score in the range [0,1] following this criteria:
-1.0 — Full alignment: The second opinion has essentially the same overall stance as the first, including the same direction, degree of support/opposition, and relevant ambivalence or qualifications. Differences in values, reasons, emphasis, or wording are minor and do not materially change the position.
-[0.7, 0.9] — Strong alignment: The second opinion substantially shares the first opinion's stance. It may differ in the strength of its position, in how it balances competing considerations, or in its values and reasons, but the main position is clearly aligned.
-[0.4, 0.6] — Partial alignment: The opinions share some meaningful aspects of their stance but also differ in important ways. This may include agreeing on one side of an issue while disagreeing on another, having different degrees of ambivalence, or reaching similar overall conclusions while placing substantially different weight on competing considerations. 
-[0.1, 0.3] — Weak alignment: The opinions largely differ in their overall stance, including the way they resolve or balance competing considerations.
-0.0 — No alignment: The opinions have fundamentally incompatible stances.

Return only the score."""
    }
}
