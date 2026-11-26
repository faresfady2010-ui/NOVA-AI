"""Bounded conversation history formatted for instruction-tuned NOVA."""
from inference.generate import generate
from training.instruction_dataset import USER_MARKER, ASSISTANT_MARKER


class Conversation:
    def __init__(self, model, tokenizer, *, max_new_tokens=160, temperature=0.8, top_k=40, device=None):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_k, self.device = temperature, top_k, device

    @staticmethod
    def format_history(messages: list[dict]) -> str:
        chunks = []
        for message in messages:
            role, content = message.get("role"), message.get("content", "")
            if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
                continue
            marker = USER_MARKER if role == "user" else ASSISTANT_MARKER
            chunks.append(marker + content.strip() + "\n")
        return "".join(chunks)

    def respond(self, messages: list[dict]) -> str:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        clean = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                 and isinstance(m.get("content"), str) and m["content"].strip()]
        if not clean or clean[-1]["role"] != "user":
            raise ValueError("Please end the conversation with a non-empty user message.")
        context_limit = self.model.config.context_length
        generation_budget = min(max(self.max_new_tokens, 1), context_limit - 1)
        prompt_budget = max(1, context_limit - generation_budget)
        # Trim whole oldest turns first, avoiding cross-conversation state and
        # preserving recent user/assistant context whenever it fits.
        while len(clean) > 1 and len(self.tokenizer.encode(self.format_history(clean))) > prompt_budget:
            clean = clean[2:] if len(clean) > 2 else clean[1:]
        prompt = self.format_history(clean)
        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > prompt_budget:
            latest = clean[-1]
            marker_ids = self.tokenizer.encode(USER_MARKER)
            content_ids = self.tokenizer.encode(latest["content"])
            allowed = max(0, prompt_budget - len(marker_ids) - len(self.tokenizer.encode("\n")))
            content = self.tokenizer.decode(content_ids[-allowed:]) if allowed else ""
            prompt = USER_MARKER + content + "\n"
        prompt += ASSISTANT_MARKER
        return generate(self.model, self.tokenizer, prompt, max_new_tokens=generation_budget,
                        temperature=self.temperature, top_k=self.top_k, device=self.device,
                        stop_sequence="\n<|end|>").strip()
