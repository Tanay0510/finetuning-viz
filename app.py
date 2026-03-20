"""
LLM Fine-Tuning Visual Studio — llmviz.studio
"""
import os
import json
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash

# ... (rest of imports and app config)

def load_constants():
    with open(os.path.join('data', 'constants.json'), 'r') as f:
        return json.load(f)

constants = load_constants()
METHODS = constants['METHODS']
GPUS = constants['GPUS']
MODEL_SIZES = constants['MODEL_SIZES']
DEEP_DIVES = constants['DEEP_DIVES']
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-llmviz-123')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('NEON_DATABASE_URL', 'sqlite:///llmviz.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── Models ──
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    recipes = db.relationship('Recipe', backref='author', lazy=True)

class Recipe(db.Model):
    __tablename__ = 'recipes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    config = db.Column(db.JSON, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Global Data (Loaded from data/constants.json) ──
# (Already loaded above)


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

@app.route("/api/config")
def api_config():
    return jsonify({
        "methods": METHODS,
        "gpus": GPUS,
        "models": MODEL_SIZES
    })

@app.route("/api/recipes", methods=['GET', 'POST'])
@login_required
def api_recipes():
    if request.method == 'POST':
        data = request.json
        new_recipe = Recipe(
            name=data.get('name', 'Untitled Recipe'),
            config=data.get('config'),
            user_id=current_user.id
        )
        db.session.add(new_recipe)
        db.session.commit()
        return jsonify({"message": "Recipe saved", "id": new_recipe.id})
    
    recipes = Recipe.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        "id": r.id,
        "name": r.name,
        "config": r.config
    } for r in recipes])

@app.route("/api/methods")
def api_methods():
    return jsonify(METHODS)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
