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
from flask_migrate import Migrate
from authlib.integrations.flask_client import OAuth
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-llmviz-123')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('NEON_DATABASE_URL', 'sqlite:///llmviz.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ── Models ──
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=True)
    name = db.Column(db.String(100), nullable=False)
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    recipes = db.relationship('Recipe', backref='author', lazy=True)

class Recipe(db.Model):
    __tablename__ = 'recipes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    config = db.Column(db.JSON, nullable=False)
    is_public = db.Column(db.Boolean, default=False)
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

@app.route("/login/google")
def login_google():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/login/google/callback")
def google_callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    if user_info:
        user = User.query.filter_by(email=user_info['email']).first()
        if not user:
            user = User(
                email=user_info['email'],
                name=user_info['name'],
                google_id=user_info['sub']
            )
            db.session.add(user)
            db.session.commit()
        else:
            if not user.google_id:
                user.google_id = user_info['sub']
                db.session.commit()
        login_user(user)
        return redirect(url_for('explore'))
    return redirect(url_for('login'))

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

@app.route("/quantize")
@login_required
def quantize():
    return render_template("quantize.html", methods=METHODS)

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
            description=data.get('description', ''),
            config=data.get('config'),
            is_public=data.get('is_public', False),
            user_id=current_user.id
        )
        db.session.add(new_recipe)
        db.session.commit()
        return jsonify({"message": "Recipe saved", "id": new_recipe.id})
    
    recipes = Recipe.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        "id": r.id,
        "name": r.name,
        "description": r.description,
        "config": r.config,
        "is_public": r.is_public
    } for r in recipes])

@app.route("/api/recipes/<int:recipe_id>")
def api_get_recipe(recipe_id):
    recipe = Recipe.query.get_or_404(recipe_id)
    # Only allow if public or if it's the owner
    if not recipe.is_public and (not current_user.is_authenticated or recipe.user_id != current_user.id):
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({
        "id": recipe.id,
        "name": recipe.name,
        "config": recipe.config
    })

@app.route("/api/recipes/delete/<int:recipe_id>", methods=['DELETE'])
@login_required
def api_delete_recipe(recipe_id):
    recipe = Recipe.query.get_or_404(recipe_id)
    if recipe.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    db.session.delete(recipe)
    db.session.commit()
    return jsonify({"message": "Recipe deleted"})

@app.route("/zoo")
def zoo():
    public_recipes = Recipe.query.filter_by(is_public=True).all()
    return render_template("zoo.html", recipes=public_recipes, methods=METHODS)

@app.route("/profile")
@login_required
def profile():
    user_recipes = Recipe.query.filter_by(user_id=current_user.id).all()
    return render_template("profile.html", recipes=user_recipes)

@app.route("/api/methods")
def api_methods():
    return jsonify(METHODS)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
