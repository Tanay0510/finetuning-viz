"""
LLM Fine-Tuning 3D Visualizer
==============================
A Flask web application that renders interactive 3D visualizations
of different LLM fine-tuning methods using Three.js.

Run:
    python app.py

Deploy:
    gunicorn app:app --bind 0.0.0.0:8000
"""

from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ─── Method data served as JSON so it's easy to edit from Python ───
METHODS = [
    {
        "id": "full",
        "name": "Full Fine-Tuning",
        "short": "Full FT",
        "emoji": "🔥",
        "color": "#FF4060",
        "accent": "#FF8095",
        "desc": (
            "Every neuron and every connection is alive — the entire network retrains. "
            "Watch all nodes glow and all paths carry gradient signals. This is the most "
            "expensive approach: for a 70B model you need ~280GB of VRAM just for training state."
        ),
        "stats": {
            "params_trained": "100%",
            "vram": "~280GB for 70B",
            "checkpoint": "Full model (~140GB)",
            "speed": "Slow (hours/days)",
        },
        "formula": "θ' = θ − α · ∇L(θ)",
        "insight": (
            "All parameters receive gradient updates via backpropagation. Adam optimizer "
            "maintains momentum and variance states for each of the billions of parameters, "
            "requiring ~4× model size in memory."
        ),
    },
    {
        "id": "lora",
        "name": "LoRA",
        "short": "LoRA",
        "emoji": "🔌",
        "color": "#3B82F6",
        "accent": "#93C5FD",
        "desc": (
            "The main network stays dark (frozen). A thin parallel bypass network glows "
            "beside it — those are the low-rank A and B matrices. Only the bypass trains. "
            "The product BA gets added to the frozen weights: W' = W₀ + BA."
        ),
        "stats": {
            "params_trained": "~0.1-2%",
            "vram": "~16GB for 7B",
            "checkpoint": "Adapter only (10-50MB)",
            "speed": "Fast",
        },
        "formula": "W' = W₀ + BA,  B∈ℝ^{d×r}, A∈ℝ^{r×d}",
        "insight": (
            "Weight updates during fine-tuning have low intrinsic rank. LoRA exploits this "
            "by learning only a rank-r decomposition. With r=16 on d=4096, each adapter pair "
            "is only 131K params vs 16.7M per original matrix."
        ),
    },
    {
        "id": "qlora",
        "name": "QLoRA",
        "short": "QLoRA",
        "emoji": "📦",
        "color": "#10B981",
        "accent": "#6EE7B7",
        "desc": (
            "Same bypass adapter as LoRA, but the frozen main network has smaller nodes — "
            "compressed to 4-bit NormalFloat. The adapter stays full BF16 precision. "
            "Paged optimizers prevent OOM by swapping states to CPU."
        ),
        "stats": {
            "params_trained": "~0.1-2%",
            "vram": "~6GB for 7B",
            "checkpoint": "Adapter only (10-50MB)",
            "speed": "Moderate (dequant overhead)",
        },
        "formula": "W₄bit + BA,  NF4 quantization + paged Adam",
        "insight": (
            "NF4 is information-theoretically optimal for normally-distributed weights. "
            "The base model only needs accurate forward pass activations — 4-bit preserves "
            "these well. Adapter gradients stay in full BF16 for stable training."
        ),
    },
    {
        "id": "prefix",
        "name": "Prefix Tuning",
        "short": "Prefix",
        "emoji": "📎",
        "color": "#F59E0B",
        "accent": "#FCD34D",
        "desc": (
            "The whole network is frozen. A cluster of bright nodes floats at the input, "
            "injecting learned signals that steer the frozen network's behavior — like a "
            "learned system prompt operating in continuous activation space."
        ),
        "stats": {
            "params_trained": "~0.1%",
            "vram": "~14GB for 7B",
            "checkpoint": "Prefix vectors (~1MB)",
            "speed": "Fast",
        },
        "formula": "P = [p₁,...,pₘ] prepended to K, V at each layer",
        "insight": (
            "Unlike discrete prompt tuning, prefix tuning operates in continuous activation "
            "space — much more expressive. The prefix vectors are trained via a reparameterization "
            "MLP for stability, then the MLP is discarded at inference."
        ),
    },
    {
        "id": "adapters",
        "name": "Adapter Layers",
        "short": "Adapters",
        "emoji": "🧩",
        "color": "#8B5CF6",
        "accent": "#C4B5FD",
        "desc": (
            "Frozen network layers with small glowing bottleneck sub-networks inserted between "
            "them. Data flows through frozen → adapter bottleneck → frozen. The residual "
            "connection ensures the adapter starts as near-identity."
        ),
        "stats": {
            "params_trained": "~0.5-3%",
            "vram": "~18GB for 7B",
            "checkpoint": "Adapter modules (~30MB)",
            "speed": "Moderate",
        },
        "formula": "h ← h + f(h·W_down)·W_up",
        "insight": (
            "The bottleneck architecture (d→m→d where m<<d) forces the adapter to learn a "
            "compressed representation. The residual connection is critical — initialized to "
            "near-zero, the adapter starts as identity and gradually adds task-specific changes."
        ),
    },
    {
        "id": "rlhf",
        "name": "RLHF / DPO",
        "short": "RLHF/DPO",
        "emoji": "👍",
        "color": "#EC4899",
        "accent": "#F9A8D4",
        "desc": (
            "The full network trains, shaped by preference signals. A green node (preferred "
            "response) pulses and grows while a red node (rejected) fades. The policy model "
            "learns to maximize human preference alignment."
        ),
        "stats": {
            "params_trained": "100%",
            "vram": "~350GB (PPO) / ~200GB (DPO)",
            "checkpoint": "Full model",
            "speed": "Very slow",
        },
        "formula": "L_DPO = −log σ(β · [log π/πref(yw) − log π/πref(yl)])",
        "insight": (
            "DPO's key insight: the optimal reward is implicitly defined by the log-ratio "
            "of the policy to reference. This eliminates the reward model entirely — you "
            "directly optimize on preference pairs without PPO's instability."
        ),
    },
]


@app.route("/")
def index():
    """Render the main visualization page."""
    return render_template("index.html", methods=METHODS)


@app.route("/api/methods")
def api_methods():
    """JSON endpoint for method data (useful for AJAX or future extensions)."""
    return jsonify(METHODS)


@app.route("/api/method/<method_id>")
def api_method(method_id):
    """JSON endpoint for a single method."""
    method = next((m for m in METHODS if m["id"] == method_id), None)
    if method is None:
        return jsonify({"error": "Method not found"}), 404
    return jsonify(method)


if __name__ == "__main__":
    print("=" * 50)
    print("  LLM Fine-Tuning 3D Visualizer")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
