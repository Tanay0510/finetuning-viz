"""
LLM Fine-Tuning Visual Studio — llmviz.studio
"""
import os
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-llmviz-123')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('NEON_DATABASE_URL', 'sqlite:///llmviz.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── User Model ──
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Global Methods Data (abbreviated for clarity) ──
METHODS = [
    {
        "id": "full", "name": "Full Fine-Tuning", "short": "Full FT", "emoji": "🔥",
        "color": "#FF4060", "accent": "#FF8095", "tagline": "Maximum power. Maximum cost.",
        "desc": "Updates every parameter via standard backpropagation.", "formula": "θ' = θ − α · ∇L(θ)",
        "insight": "For a 70B model in BF16: ~560GB minimum VRAM.", "stats": {"params": "100%", "vram_7b": "~60GB", "vram_70b": "~560GB", "checkpoint": "Full model", "speed": "Slow"},
        "paper": {}, "code": "..."
    },
    {
        "id": "lora", "name": "LoRA", "short": "LoRA", "emoji": "🔌",
        "color": "#3B82F6", "accent": "#93C5FD", "tagline": "Low-rank adapters. 1000x smaller checkpoints.",
        "desc": "Freezes all pretrained weights, injects trainable low-rank matrices A and B.", "formula": "W' = W₀ + BA",
        "insight": "0.8% of parameters, ~95% of full FT quality.", "stats": {"params": "~0.1-2%", "vram_7b": "~16GB", "vram_70b": "~80GB", "checkpoint": "10-50MB", "speed": "Fast"},
        "paper": {}, "code": "..."
    },
    {
        "id": "qlora", "name": "QLoRA", "short": "QLoRA", "emoji": "📦", "color": "#10B981", "accent": "#6EE7B7", "tagline": "4-bit base + full-precision adapters. 65B on one GPU.", "desc": "NF4 quantization of frozen weights + LoRA adapters in BF16.", "formula": "W_NF4 + BA", "insight": "65B model = ~33GB in NF4. Fine-tune on a single 48GB GPU.", "stats": {"params": "~0.1-2%", "vram_7b": "~6GB", "vram_70b": "~36GB", "checkpoint": "10-50MB", "speed": "Moderate"}, "paper": {}, "code": "..." },
    {
        "id": "dora", "name": "DoRA", "short": "DoRA", "emoji": "⚖️", "color": "#A78BFA", "accent": "#C4B5FD", "tagline": "Weight-Decomposed Low-Rank Adaptation.", "desc": "Decomposes weight updates into magnitude (m) and direction (V).", "formula": "W = m · (W₀ + BA) / ||W₀ + BA||_f", "insight": "Achieves higher learning capacity than LoRA without increasing inference latency.", "stats": {"params": "~0.1-2%", "vram_7b": "~16GB", "vram_70b": "~80GB", "checkpoint": "15-60MB", "speed": "Moderate"}, "paper": {}, "code": "..." },
    {
        "id": "galore", "name": "GaLore", "short": "GaLore", "emoji": "🌌", "color": "#F472B6", "accent": "#F9A8D4", "tagline": "Full-parameter learning in low-rank subspaces.", "desc": "Projects gradients into a low-rank space using SVD.", "formula": "G = P · G_low · Qᵀ", "insight": "Reduces optimizer memory by up to 65.5%, enabling 7B full-parameter training on a 24GB GPU.", "stats": {"params": "100%", "vram_7b": "~22GB", "vram_70b": "~180GB", "checkpoint": "Full model", "speed": "Moderate"}, "paper": {}, "code": "..." },
    {
        "id": "prefix", "name": "Prefix Tuning", "short": "Prefix", "emoji": "📎", "color": "#F59E0B", "accent": "#FCD34D", "tagline": "Learned soft prompts steering frozen attention.", "desc": "Prepends trainable continuous vectors to K and V at every layer.", "formula": "P = [p₁,...,pₘ] → K, V", "insight": "Swap prefixes to swap tasks. 0.075% of a 7B model.", "stats": {"params": "~0.05-0.1%", "vram_7b": "~14GB", "vram_70b": "~140GB", "checkpoint": "~1MB", "speed": "Fast"}, "paper": {}, "code": "..." },
    {
        "id": "adapters", "name": "Adapter Layers", "short": "Adapters", "emoji": "🧩", "color": "#8B5CF6", "accent": "#C4B5FD", "tagline": "Bottleneck modules between frozen layers.", "desc": "Inserts down-project → nonlinearity → up-project + residual modules.", "formula": "h ← h + f(h · W_down) · W_up", "insight": "Composable via AdapterFusion.", "stats": {"params": "~0.5-3%", "vram_7b": "~18GB", "vram_70b": "~160GB", "checkpoint": "~30MB", "speed": "Moderate"}, "paper": {}, "code": "..." },
    {
        "id": "rlhf", "name": "RLHF / DPO", "short": "RLHF/DPO", "emoji": "👍", "color": "#EC4899", "accent": "#F9A8D4", "tagline": "Human preferences sculpt model behavior.", "desc": "Aligns outputs with human preferences. DPO is simpler, more stable.", "formula": "L_DPO", "insight": "DPO eliminates the reward model by implicitly defining it.", "stats": {"params": "100%", "vram_7b": "~60GB (DPO)", "vram_70b": "~400GB+", "checkpoint": "Full model", "speed": "Very slow"}, "paper": {}, "code": "..." }
]
DEEP_DIVES = { "full": {}, "lora": {}, "qlora": {}, "dora": {}, "galore": {}, "prefix": {}, "adapters": {}, "rlhf": {} } # Abbreviated
GPUS = [{"id": "rtx3090", "name": "RTX 3090", "vram": 24}] # Abbreviated
MODEL_SIZES = [{"id": "7b", "name": "7B", "params": 7.0}] # Abbreviated

# ── Routes ──
@app.route("/")
def landing():
    return render_template("landing.html")

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
        if User.query.filter_by(email=request.form.get('email')).first():
            flash('Email address already exists.')
            return redirect(url_for('register'))
        new_user = User(
            email=request.form.get('email'),
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
    return render_template("data.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
