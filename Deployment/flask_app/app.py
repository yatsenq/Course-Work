from pathlib import Path
import os

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from ml_service import predict_news_bundle


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-in-production")

instance_path = Path(__file__).resolve().parent / "instance"
instance_path.mkdir(parents=True, exist_ok=True)
default_db = instance_path / "site.db"

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", f"sqlite:///{default_db.as_posix()}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", str(Path(__file__).resolve().parents[2]))

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Увійдіть, щоб продовжити."


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Усі поля обов'язкові.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Пароль має містити мінімум 6 символів.", "error")
            return redirect(url_for("register"))

        existing = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing:
            flash("Користувач з таким логіном або email вже існує.", "error")
            return redirect(url_for("register"))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()

        flash("Реєстрація успішна. Тепер увійдіть у систему.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Вхід виконано успішно.", "success")
            return redirect(url_for("dashboard"))

        flash("Невірний email або пароль.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Ви вийшли з акаунта.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if not text:
            flash("Введіть текст новини.", "error")
        else:
            try:
                result = predict_news_bundle(text, MODEL_BASE_DIR)
            except Exception as exc:
                flash(f"Помилка аналізу: {exc}", "error")

    return render_template("dashboard.html", result=result, text=text)


@app.cli.command("init-db")
def init_db_command():
    db.create_all()
    print("Database initialized")


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
