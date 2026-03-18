"""
LLM Fine-Tuning Visual Studio — llmviz.studio
"""
import os
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-llmviz-123')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('NEON_DATABASE_URL', 'sqlite:///llmviz.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── User Model ──
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Global Data ──
METHODS = [
    {
        "id": "full", "name": "Full Fine-Tuning", "short": "Full FT", "emoji": "🔥",
        "color": "#FF4060", "accent": "#FF8095",
        "tagline": "Maximum power. Maximum cost.",
        "desc": "Updates every parameter via standard backpropagation. Adam maintains momentum+variance for all billions of parameters, requiring ~4x model size in memory.",
        "formula": "θ' = θ − α · ∇L(θ)",
        "insight": "For a 70B model in BF16: 140GB weights + 140GB gradients + 280GB optimizer states = ~560GB minimum VRAM.",
        "stats": {"params": "100%", "vram_7b": "60", "vram_70b": "560", "checkpoint": "Full model", "speed": "Slow"},
        "paper": {"title": "Standard backpropagation", "url": None, "year": None},
        "code": "from transformers import AutoModelForCausalLM, Trainer, TrainingArguments\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\ntrainer = Trainer(\n    model=model,\n    args=TrainingArguments(\n        output_dir=\"./full-ft\",\n        per_device_train_batch_size=1,\n        gradient_accumulation_steps=16,\n        learning_rate=2e-5,\n        bf16=True,\n    ),\n    train_dataset=dataset,\n)\ntrainer.train()",
    },
    {
        "id": "lora", "name": "LoRA", "short": "LoRA", "emoji": "🔌",
        "color": "#3B82F6", "accent": "#93C5FD",
        "tagline": "Low-rank adapters. 1000x smaller checkpoints.",
        "desc": "Freezes all pretrained weights, injects trainable low-rank matrices A and B into attention layers. W' = W₀ + BA where rank r << d.",
        "formula": "W' = W₀ + BA, B∈ℝ^{d×r}, A∈ℝ^{r×d}",
        "insight": "With r=16 on d=4096, each adapter pair is 131K params vs 16.7M original — 0.8% of parameters, ~95% of full FT quality.",
        "stats": {"params": "~0.1-2%", "vram_7b": "16", "vram_70b": "80", "checkpoint": "10-50MB", "speed": "Fast"},
        "paper": {"title": "LoRA: Low-Rank Adaptation of LLMs", "url": "https://arxiv.org/abs/2106.09685", "year": 2021},
        "code": "from peft import LoraConfig, get_peft_model\nfrom transformers import AutoModelForCausalLM\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\nconfig = LoraConfig(\n    r=16,\n    lora_alpha=32,\n    target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"],\n    lora_dropout=0.05,\n    task_type=\"CAUSAL_LM\",\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()\n# trainable: 4,194,304 || total: 6,742,609,920 || 0.062%",
    },
    {
        "id": "qlora", "name": "QLoRA", "short": "QLoRA", "emoji": "📦",
        "color": "#10B981", "accent": "#6EE7B7",
        "tagline": "4-bit base + full-precision adapters. 65B on one GPU.",
        "desc": "NF4 quantization of frozen weights + LoRA adapters in BF16. Paged optimizers swap states to CPU on memory pressure.",
        "formula": "W_NF4 + BA, paged AdamW",
        "insight": "65B model = ~33GB in NF4 + ~50MB adapters. Fine-tune on a single 48GB GPU. Quality gap vs LoRA: typically <1 point.",
        "stats": {"params": "~0.1-2%", "vram_7b": "6", "vram_70b": "36", "checkpoint": "10-50MB", "speed": "Moderate"},
        "paper": {"title": "QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "year": 2023},
        "code": "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\nfrom peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\nbnb_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type=\"nf4\",\n    bnb_4bit_compute_dtype=\"bfloat16\",\n    bnb_4bit_use_double_quant=True,\n)\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"meta-llama/Llama-2-70b-hf\",\n    quantization_config=bnb_config,\n    device_map=\"auto\",\n)\nmodel = prepare_model_for_kbit_training(model)\nmodel = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=\"all-linear\"))",
    },
    {
        "id": "dora", "name": "DoRA", "short": "DoRA", "emoji": "⚖️",
        "color": "#A78BFA", "accent": "#C4B5FD",
        "tagline": "Weight-Decomposed Low-Rank Adaptation.",
        "desc": "Decomposes weight updates into magnitude (m) and direction (V). Applies LoRA to the direction component only, better mimicking full fine-tuning dynamics.",
        "formula": "W = m · (W₀ + BA) / ||W₀ + BA||_f",
        "insight": "By decoupling magnitude and direction, DoRA achieves higher learning capacity than LoRA without increasing inference latency.",
        "stats": {"params": "~0.1-2%", "vram_7b": "16", "vram_70b": "80", "checkpoint": "15-60MB", "speed": "Moderate"},
        "paper": {"title": "DoRA: Weight-Decomposed Low-Rank Adaptation", "url": "https://arxiv.org/abs/2402.09353", "year": 2024},
        "code": "from peft import LoraConfig, get_peft_model\n\nconfig = LoraConfig(\n    r=16,\n    lora_alpha=32,\n    target_modules=[\"q_proj\", \"v_proj\"],\n    use_dora=True,\n    task_type=\"CAUSAL_LM\",\n)\nmodel = get_peft_model(base_model, config)",
    },
    {
        "id": "galore", "name": "GaLore", "short": "GaLore", "emoji": "🌌",
        "color": "#F472B6", "accent": "#F9A8D4",
        "tagline": "Full-parameter learning in low-rank subspaces.",
        "desc": "Projects gradients into a low-rank space using SVD decomposition. Allows updating all model parameters while only storing low-rank optimizer states.",
        "formula": "G = P · G_low · Qᵀ, ∇θ = G",
        "insight": "Reduces optimizer memory by up to 65.5% for 7B models, enabling 7B full-parameter training on a single 24GB consumer GPU.",
        "stats": {"params": "100%", "vram_7b": "22", "vram_70b": "180", "checkpoint": "Full model", "speed": "Moderate"},
        "paper": {"title": "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", "url": "https://arxiv.org/abs/2403.03507", "year": 2024},
        "code": "from galore_torch import GaLoreAdamW8bit\n\n# Define GaLore parameters for optimizer\noptimizer_params = [\n    {'params': model.parameters(), 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25}\n]\noptimizer = GaLoreAdamW8bit(optimizer_params, lr=1e-5)",
    },
    {
        "id": "prefix", "name": "Prefix Tuning", "short": "Prefix", "emoji": "📎",
        "color": "#F59E0B", "accent": "#FCD34D",
        "tagline": "Learned soft prompts steering frozen attention.",
        "desc": "Prepends trainable continuous vectors to K and V at every layer. All model weights frozen. Like a learned system prompt in continuous space.",
        "formula": "P = [p₁,...,pₘ] → K, V at each layer",
        "insight": "m=20 prefix tokens × 32 layers × 2(K,V) × 4096 dim = 5.2M params. That's 0.075% of a 7B model. Swap prefixes to swap tasks.",
        "stats": {"params": "~0.05-0.1%", "vram_7b": "14", "vram_70b": "140", "checkpoint": "~1MB", "speed": "Fast"},
        "paper": {"title": "Prefix-Tuning: Optimizing Continuous Prompts", "url": "https://arxiv.org/abs/2101.00190", "year": 2021},
        "code": "from peft import PrefixTuningConfig, get_peft_model\nfrom transformers import AutoModelForCausalLM\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nconfig = PrefixTuningConfig(\n    task_type=\"CAUSAL_LM\",\n    num_virtual_tokens=20,\n    prefix_projection=True,\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()\n# trainable: 9,872,384 || 0.146%",
    },
    {
        "id": "adapters", "name": "Adapter Layers", "short": "Adapters", "emoji": "🧩",
        "color": "#8B5CF6", "accent": "#C4B5FD",
        "tagline": "Bottleneck modules between frozen layers.",
        "desc": "Inserts down-project → nonlinearity → up-project + residual modules after each sublayer. Starts as near-identity, gradually learns task-specific changes.",
        "formula": "h ← h + f(h · W_down) · W_up",
        "insight": "Bottleneck m=64 on d=4096: each adapter = 524K params. 2 per layer × 32 layers = 33.5M trainable. Composable via AdapterFusion.",
        "stats": {"params": "~0.5-3%", "vram_7b": "18", "vram_70b": "160", "checkpoint": "~30MB", "speed": "Moderate"},
        "paper": {"title": "Parameter-Efficient Transfer Learning for NLP", "url": "https://arxiv.org/abs/1902.00751", "year": 2019},
        "code": "from adapters import AutoAdapterModel\n\nmodel = AutoAdapterModel.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nmodel.add_adapter(\"my_task\", config=\"seq_bn\")  # bottleneck adapter\nmodel.train_adapter(\"my_task\")  # freeze base, train adapter only\n\ntrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\nprint(f\"Trainable: {trainable:,}\")",
    },
    {
        "id": "rlhf", "name": "RLHF / DPO", "short": "RLHF/DPO", "emoji": "👍",
        "color": "#EC4899", "accent": "#F9A8D4",
        "tagline": "Human preferences sculpt model behavior.",
        "desc": "Aligns outputs with human preferences. RLHF: reward model + PPO. DPO: directly optimize on preference pairs, no reward model needed.",
        "formula": "L = −log σ(β·[log π/π_ref(y_w) − log π/π_ref(y_l)])",
        "insight": "DPO eliminates the reward model — the optimal reward is implicitly defined by the policy/reference log-ratio. Simpler, more stable, less memory.",
        "stats": {"params": "100%", "vram_7b": "60", "vram_70b": "400", "checkpoint": "Full model", "speed": "Very slow"},
        "paper": {"title": "DPO: Your LM is Secretly a Reward Model", "url": "https://arxiv.org/abs/2305.18290", "year": 2023},
        "code": "from trl import DPOTrainer, DPOConfig\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nref_model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\ntokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\ntrainer = DPOTrainer(\n    model=model,\n    ref_model=ref_model,\n    args=DPOConfig(beta=0.1, learning_rate=5e-7, bf16=True),\n    train_dataset=preference_data,  # needs: prompt, chosen, rejected\n    tokenizer=tokenizer,\n)\ntrainer.train()",
    },
]

GPUS = [
    {"id": "rtx3090", "name": "RTX 3090", "vram": 24, "tflops": 35.6, "cost": 0, "tier": "Consumer"},
    {"id": "rtx4090", "name": "RTX 4090", "vram": 24, "tflops": 82.6, "cost": 0, "tier": "Consumer"},
    {"id": "a6000",   "name": "A6000",     "vram": 48, "tflops": 38.7, "cost": 0.80, "tier": "Pro"},
    {"id": "a100_40", "name": "A100 40GB", "vram": 40, "tflops": 312,  "cost": 1.10, "tier": "Cloud"},
    {"id": "a100_80", "name": "A100 80GB", "vram": 80, "tflops": 312,  "cost": 1.60, "tier": "Cloud"},
    {"id": "h100",    "name": "H100 80GB", "vram": 80, "tflops": 989,  "cost": 2.50, "tier": "Cloud"},
    {"id": "4xa100",  "name": "4×A100 80G","vram": 320,"tflops": 1248, "cost": 6.40, "tier": "Multi"},
    {"id": "8xh100",  "name": "8×H100",   "vram": 640,"tflops": 7912, "cost": 20.0, "tier": "Multi"},
]

MODEL_SIZES = [
    {"id": "1b",  "name": "1.3B", "params": 1.3, "d": 2048, "layers": 24},
    {"id": "3b",  "name": "3B",   "params": 3.0, "d": 3200, "layers": 26},
    {"id": "7b",  "name": "7B",   "params": 7.0, "d": 4096, "layers": 32},
    {"id": "13b", "name": "13B",  "params": 13.0,"d": 5120, "layers": 40},
    {"id": "33b", "name": "33B",  "params": 33.0,"d": 6656, "layers": 52},
    {"id": "65b", "name": "65B",  "params": 65.0,"d": 8192, "layers": 64},
    {"id": "70b", "name": "70B",  "params": 70.0,"d": 8192, "layers": 80},
]

DEEP_DIVES = {
    "full": {
        "sections": [
            {"title": "How It Works", "content": "Full fine-tuning updates every parameter via standard backpropagation."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "lora": {
        "sections": [
            {"title": "Low-Rank Adaptation", "content": "Freezes base weights, trains tiny A & B matrices."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "qlora": {
        "sections": [
            {"title": "Quantized LoRA", "content": "NF4 quantization + LoRA adapters."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "dora": {
        "sections": [
            {"title": "Weight Decomposition", "content": "Separates magnitude and direction of weights."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "galore": {
        "sections": [
            {"title": "Gradient Projection", "content": "Projects gradients into low-rank space."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "prefix": {
        "sections": [
            {"title": "Prefix Tuning", "content": "Learns virtual tokens at every layer."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "adapters": {
        "sections": [
            {"title": "Adapter Layers", "content": "Inserts bottleneck modules between layers."}
        ],
        "benchmarks": [],
        "tips": []
    },
    "rlhf": {
        "sections": [
            {"title": "DPO Alignment", "content": "Aligns model with human preference pairs."}
        ],
        "benchmarks": [],
        "tips": []
    },
}

# ── Routes ──
@app.route("/")
def landing():
    return render_template("landing.html", methods=METHODS)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('explore'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('explore'))
        flash('Invalid credentials. Please try again.')
    return render_template("login.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('explore'))
    if request.method == 'POST':
        email = request.form.get('email')
        if User.query.filter_by(email=email).first():
            flash('Email address already exists.')
            return redirect(url_for('register'))
        new_user = User(
            email=email,
            name=request.form.get('name'),
            password=generate_password_hash(request.form.get('password'), method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('explore'))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route("/explore")
@login_required
def explore():
    return render_template("explore.html", methods=METHODS)

@app.route("/playground")
@login_required
def playground():
    return render_template("playground.html", methods=METHODS, gpus=GPUS, models=MODEL_SIZES)

@app.route("/learn/<method_id>")
@login_required
def learn(method_id):
    method = next((m for m in METHODS if m["id"] == method_id), None)
    if method is None: return "Method not found", 404
    deep = DEEP_DIVES.get(method_id, {})
    return render_template("learn.html", method=method, deep=deep, methods=METHODS)

@app.route("/guide")
@login_required
def guide():
    return render_template("guide.html", methods=METHODS)

@app.route("/data")
@login_required
def data_prep():
    return render_template("data.html", methods=METHODS)

@app.route("/compare")
@login_required
def compare():
    return render_template("compare.html", methods=METHODS)

@app.route("/attention")
@login_required
def attention():
    return render_template("attention.html", methods=METHODS)

@app.route("/api/methods")
def api_methods():
    return jsonify(METHODS)

@app.route("/chat", methods=['POST'])
@login_required
def chat():
    user_message = request.json.get("message")
    
    system_prompt = """
    You are the 'AI Scene Director' for llmviz.studio, a 3D visualization platform for LLM fine-tuning.
    Your goal is to explain technical concepts while simultaneously controlling the 3D viewport.
    
    You MUST respond in valid JSON format only.
    Response Structure:
    {
        "text": "Your verbal explanation here",
        "commands": [
            { "action": "SWITCH_METHOD", "value": "lora" },
            { "action": "TOGGLE_EXPLODE", "value": true },
            { "action": "TOGGLE_CLUSTER", "value": false },
            { "action": "SET_TIMELINE", "value": 0.5 }
        ]
    }
    
    Available Methods: lora, full, qlora, dora, galore, prefix, adapters, rlhf.
    Timeline: 0.0 (fresh) to 1.0 (trained).
    
    Instructions:
    - If they ask to see a method (e.g., 'Show me LoRA'), use SWITCH_METHOD.
    - If they want to see internal layers, use TOGGLE_EXPLODE.
    - If they ask about scaling or multiple GPUs, use TOGGLE_CLUSTER.
    - If they ask to see the training effect, use SET_TIMELINE.
    - Be technical, concise, and helpful.
    """

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={ "type": "json_object" }
        )
        return completion.choices[0].message.content
    except Exception as e:
        return jsonify({"text": "System error: " + str(e), "commands": []}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
