"""NOVA's local Flask API. No external model APIs or pretrained weights are used."""
from __future__ import annotations
import os
from flask import Flask, jsonify, request, send_from_directory
from inference.conversation import Conversation
from inference.generate import load_model

app = Flask(__name__, static_folder="../frontend", static_url_path="")
_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        path = os.environ.get("NOVA_CHECKPOINT", "checkpoints/nova.pt")
        if not os.path.exists(path): raise FileNotFoundError(f"No trained checkpoint at {path}. Train NOVA first.")
        model, tokenizer, _, device = load_model(path)
        _runtime = (model, tokenizer, device)
    return _runtime


@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _runtime is not None, "experimental": True})

@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages", [])
    if not isinstance(messages, list): return jsonify({"error": "messages must be a list"}), 400
    if not any(isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip() for m in messages):
        return jsonify({"error": "Message cannot be empty."}), 400
    try:
        model, tokenizer, device = get_runtime()
        temperature = float(payload.get("temperature", 0.8)); max_tokens = int(payload.get("max_new_tokens", 160))
        if not 0.1 <= temperature <= 2.0: return jsonify({"error": "temperature must be between 0.1 and 2.0"}), 400
        if not 1 <= max_tokens <= 512: return jsonify({"error": "max_new_tokens must be between 1 and 512"}), 400
        conversation = Conversation(model, tokenizer, max_new_tokens=max_tokens, temperature=temperature, device=device)
        return jsonify({"reply": conversation.respond(messages), "experimental": True})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("NOVA inference failed")
        return jsonify({"error": "NOVA could not generate a response. Check the server logs."}), 500

if __name__ == "__main__": app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
