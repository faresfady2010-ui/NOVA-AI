"""Conversation formatting and model-backed response generation."""
from inference.generate import generate


class Conversation:
    def __init__(self, model, tokenizer, *, max_new_tokens=160, temperature=0.8, top_k=40, device=None):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature, self.top_k, self.device = max_new_tokens, temperature, top_k, device

    @staticmethod
    def format_history(messages: list[dict]) -> str:
        lines=[]
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                lines.append(f"{role.upper()}: {content.strip()}")
        return "\n".join(lines)

    def respond(self, messages: list[dict]) -> str:
        prompt = self.format_history(messages)
        if not prompt.strip(): raise ValueError("Please enter a message before sending.")
        # Keep recent conversation context within the model's byte-based window.
        max_context = self.model.config.context_length - min(self.max_new_tokens, self.model.config.context_length - 1)
        encoded = self.tokenizer.encode(prompt)
        if len(encoded) > max_context:
            prompt = self.tokenizer.decode(encoded[-max_context:])
        prompt += "\nASSISTANT:"
        return generate(self.model, self.tokenizer, prompt, max_new_tokens=min(self.max_new_tokens, self.model.config.context_length), temperature=self.temperature, top_k=self.top_k, device=self.device).strip()
