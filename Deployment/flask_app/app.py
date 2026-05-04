from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import threading
import hashlib
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for, Response
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import or_
from werkzeug.security import check_password_hash, generate_password_hash

import ml_service


def normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif raw_url.startswith("postgresql://"):
        raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    if raw_url.startswith("postgresql+") and "sslmode=" not in raw_url:
        separator = "&" if "?" in raw_url else "?"
        raw_url = f"{raw_url}{separator}sslmode=require"

    return raw_url


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def is_valid_email(value: str) -> bool:
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(email_pattern, value.strip()))


@lru_cache(maxsize=1)
def get_pdf_fonts() -> tuple[str, str]:
    base_dir = Path(__file__).resolve().parent
    regular_candidates = [
        base_dir / "static" / "fonts" / "DejaVuSans.ttf",
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        base_dir / "static" / "fonts" / "DejaVuSans-Bold.ttf",
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]

    regular_font = next((path for path in regular_candidates if path.exists()), None)
    bold_font = next((path for path in bold_candidates if path.exists()), None)

    if regular_font and bold_font:
        try:
            pdfmetrics.registerFont(TTFont("AppSans", str(regular_font)))
            pdfmetrics.registerFont(TTFont("AppSans-Bold", str(bold_font)))
            return "AppSans", "AppSans-Bold"
        except Exception:
            pass

    return "Helvetica", "Helvetica-Bold"


app = Flask(__name__)

base_dir = Path(__file__).resolve().parent
load_env_file(base_dir / ".env")

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    secret_key = secrets.token_urlsafe(48)
app.config["SECRET_KEY"] = secret_key

instance_path = base_dir / "instance"
default_db = instance_path / "site.db"

database_url = os.getenv("DATABASE_URL", "").strip()
if database_url:
    database_url = normalize_database_url(database_url)
    if database_url.startswith("postgresql+") and default_db.exists():
        try:
            default_db.unlink()
        except OSError:
            pass
else:
    instance_path.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{default_db.as_posix()}"

app.config["SQLALCHEMY_DATABASE_URI"] = normalize_database_url(database_url)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

def get_model_base_dir():
    # 1. Environment variable
    env_val = os.getenv("MODEL_BASE_DIR")
    if env_val:
        return env_val
    
    # 2. Look in current directory (if models are uploaded to same root)
    current = Path(__file__).resolve().parent
    if (current / "Models").exists():
        return str(current)
        
    # 3. Look two levels up (original local structure: Course Work/Deployment/flask_app/app.py)
    two_up = current.parents[1] if len(current.parents) > 1 else current
    if (two_up / "Models").exists():
        return str(two_up)
    
    # 4. Fallback to current
    return str(current)

MODEL_BASE_DIR = get_model_base_dir()

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Увійдіть, щоб продовжити."


EXAMPLE_NEWS = [
    {
        "title": "Приклад 1: політика",
        "text": "Уряд оголосив нову програму підтримки громад, яка має підсилити місцеві бюджети та інфраструктурні проєкти.",
    },
    {
        "title": "Приклад 2: спорт",
        "text": "Команда несподівано перемогла у фіналі чемпіонату, а тренер відзначив дисципліну та командну роботу гравців.",
    },
    {
        "title": "Приклад 3: фейк-стиль",
        "text": "Неймовірна сенсація: вчені терміново підтвердили подію, про яку всі мовчали, а джерела розкриють усе вже сьогодні.",
    },
]


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class CheckHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    input_text = db.Column(db.Text, nullable=False)
    fake_label = db.Column(db.String(8), nullable=False)
    prob_true = db.Column(db.Float, nullable=False)
    prob_fake = db.Column(db.Float, nullable=False)
    theme = db.Column(db.String(120), nullable=False)
    theme_confidence = db.Column(db.Float, nullable=False)
    theme_markers_json = db.Column(db.Text, nullable=False)
    analysis_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    @property
    def theme_markers(self):
        return json.loads(self.theme_markers_json or "[]")

    @property
    def analysis(self):
        return json.loads(self.analysis_json or "{}")

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    history_id = db.Column(db.Integer, db.ForeignKey("check_history.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

class AnalysisCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text_hash = db.Column(db.String(64), unique=True, nullable=False)
    result_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def save_history_item(user_id: int, text: str, result: dict) -> CheckHistory:
    history = CheckHistory(
        user_id=user_id,
        input_text=text,
        fake_label=result["fake_label"],
        prob_true=result["prob_true"],
        prob_fake=result["prob_fake"],
        theme=result["theme"],
        theme_confidence=result["theme_confidence"],
        theme_markers_json=json.dumps(result.get("theme_markers", []), ensure_ascii=False),
        analysis_json=json.dumps({**result, "input_text": text}, ensure_ascii=False),
    )
    db.session.add(history)
    db.session.commit()
    return history


def _pretty_model_name(model_name: str) -> str:
    return model_name.replace("_", " ").strip()


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_value(raw_value: object) -> float:
    try:
        if raw_value in {None, ""}:
            return 0.0
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


@lru_cache(maxsize=1)
def load_model_comparison_metrics() -> dict[str, dict[str, object]]:
    results_dir = Path(MODEL_BASE_DIR) / "experiments" / "results"

    fake_rows: list[dict[str, object]] = []
    baseline_path = results_dir / "01_baseline_fake_metrics.json"
    if baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_metrics = baseline_payload.get("metrics", {})
        fake_rows.append(
            {
                "model": _pretty_model_name(str(baseline_payload.get("model_name", "Baseline_LogReg"))),
                "accuracy": _float_value(baseline_metrics.get("accuracy")),
                "precision": _float_value(baseline_metrics.get("precision")),
                "recall": _float_value(baseline_metrics.get("recall")),
                "f1": _float_value(baseline_metrics.get("f1")),
                "roc_auc": _float_value(baseline_metrics.get("roc_auc")),
            }
        )

    for row in _read_csv_rows(results_dir / "08_fake_detection_summary.csv"):
        fake_rows.append(
            {
                "model": _pretty_model_name(row.get("", "")),
                "accuracy": _float_value(row.get("Accuracy")),
                "precision": _float_value(row.get("Precision")),
                "recall": _float_value(row.get("Recall")),
                "f1": _float_value(row.get("F1-Score")),
                "roc_auc": _float_value(row.get("ROC-AUC")),
            }
        )

    best_fake = max(fake_rows, key=lambda item: (item["f1"], item["roc_auc"])) if fake_rows else None
    fake_detection = {
        "title": "Fake detection",
        "models_compared": len(fake_rows),
        "best_overall": best_fake["model"] if best_fake else "—",
        "best_metric_label": "F1",
        "best_metric_value": _format_percent(float(best_fake["f1"])) if best_fake else "—",
        "best_secondary_label": "ROC-AUC",
        "best_secondary_value": _format_percent(float(best_fake["roc_auc"])) if best_fake else "—",
        "rows": [
            {
                "model": row["model"],
                "accuracy": _format_percent(float(row["accuracy"])),
                "precision": _format_percent(float(row["precision"])),
                "recall": _format_percent(float(row["recall"])),
                "f1": _format_percent(float(row["f1"])),
                "roc_auc": _format_percent(float(row["roc_auc"])),
                "is_best": bool(best_fake and row["model"] == best_fake["model"]),
            }
            for row in fake_rows
        ],
    }

    topic_rows: list[dict[str, object]] = []
    for row in _read_csv_rows(results_dir / "08_topic_classification_summary.csv"):
        topic_rows.append(
            {
                "model": _pretty_model_name(row.get("", "")),
                "accuracy": _float_value(row.get("Accuracy")),
                "macro_f1": _float_value(row.get("Macro-F1")),
                "weighted_f1": _float_value(row.get("Weighted-F1")),
            }
        )

    best_topic = max(topic_rows, key=lambda item: (item["macro_f1"], item["accuracy"])) if topic_rows else None
    topic_classification = {
        "title": "Topic classification",
        "models_compared": len(topic_rows),
        "best_overall": best_topic["model"] if best_topic else "—",
        "best_metric_label": "Macro-F1",
        "best_metric_value": _format_percent(float(best_topic["macro_f1"])) if best_topic else "—",
        "best_secondary_label": "Accuracy",
        "best_secondary_value": _format_percent(float(best_topic["accuracy"])) if best_topic else "—",
        "rows": [
            {
                "model": row["model"],
                "accuracy": _format_percent(float(row["accuracy"])),
                "macro_f1": _format_percent(float(row["macro_f1"])),
                "weighted_f1": _format_percent(float(row["weighted_f1"])),
                "is_best": bool(best_topic and row["model"] == best_topic["model"]),
            }
            for row in topic_rows
        ],
    }

    return {
        "fake_detection": fake_detection,
        "topic_classification": topic_classification,
    }


def get_history_items(limit: int | None = 12):
    if not current_user.is_authenticated:
        return []

    query = CheckHistory.query.filter_by(user_id=current_user.id).order_by(CheckHistory.created_at.desc())
    if isinstance(limit, int) and limit > 0:
        query = query.limit(limit)
    return query.all()


def build_pdf_buffer(payload: dict) -> io.BytesIO:
    body_font, bold_font = get_pdf_fonts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0f6b58"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4b5563"),
    )

    def para(text: str, style=body_style):
        return Paragraph(escape(text).replace("\n", "<br/>"), style)

    story = [
        Paragraph("Звіт аналізу новини", title_style),
        para(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", small_style),
        Spacer(1, 6),
        Paragraph("Вхідний текст", section_style),
        para(payload.get("input_text", "")),
        Paragraph("Результат fake detection", section_style),
    ]

    fake_rows = [
        ["Мітка", payload.get("fake_label", "—")],
        ["Ймовірність TRUE", f"{payload.get('prob_true', 0) * 100:.2f}%"],
        ["Ймовірність FAKE", f"{payload.get('prob_fake', 0) * 100:.2f}%"],
    ]
    fake_table = Table(fake_rows, colWidths=[55 * mm, 90 * mm])
    fake_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )
    story.append(fake_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Тематична класифікація", section_style))
    topic_rows = [
        ["Категорія", payload.get("theme", "—")],
        ["Впевненість", f"{payload.get('theme_confidence', 0) * 100:.2f}%"],
    ]
    topic_table = Table(topic_rows, colWidths=[55 * mm, 90 * mm])
    topic_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ]
        )
    )
    story.append(topic_table)
    story.append(Spacer(1, 8))

    markers = payload.get("theme_markers") or []
    story.append(Paragraph("Ключові слова-маркери", section_style))
    if markers:
        marker_rows = [["Слово", "Вага"]]
        for marker in markers:
            marker_rows.append([marker.get("keyword", ""), f"{marker.get('score', 0):.4f}"])
        table = Table(marker_rows, colWidths=[95 * mm, 50 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d1fae5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#064e3b")),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                    ("FONTNAME", (0, 1), (-1, -1), body_font),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 11),
                ]
            )
        )
        story.append(table)
    else:
        story.append(para("Для цієї теми маркери не вдалося витягнути з моделі.", small_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.context_processor
def inject_globals():
    stats = None
    if current_user.is_authenticated:
        items = CheckHistory.query.filter_by(user_id=current_user.id).all()
        total = len(items)
        today = datetime.utcnow().date()
        today_count = sum(1 for i in items if i.created_at.date() == today)
        true_count = sum(1 for i in items if i.fake_label == "TRUE")
        fake_count = total - true_count
        themes = {}
        for i in items:
            themes[i.theme] = themes.get(i.theme, 0) + 1
        top_theme = max(themes, key=themes.get) if themes else "—"
        
        # Extended chart data
        theme_labels = list(themes.keys())
        theme_data = list(themes.values())
        
        # Global daily stats
        global_today_fakes = CheckHistory.query.filter(
            CheckHistory.fake_label == "FAKE",
            db.func.date(CheckHistory.created_at) == today
        ).count()

        stats = {
            "total": total,
            "today_count": today_count,
            "true_count": true_count,
            "fake_count": fake_count,
            "top_theme": top_theme,
            "theme_labels": theme_labels,
            "theme_data": theme_data,
            "global_today_fakes": global_today_fakes,
        }
    return {
        "examples": EXAMPLE_NEWS,
        "guest_mode": session.get("guest_mode", False) and not current_user.is_authenticated,
        "model_metrics": load_model_comparison_metrics(),
        "stats": stats,
    }


_model_preload_started = False
_model_preload_lock = threading.Lock()


def ensure_model_preload() -> None:
    global _model_preload_started

    with _model_preload_lock:
        if _model_preload_started:
            return
        _model_preload_started = True

    def _load() -> None:
        try:
            ml_service.load_selected_models(MODEL_BASE_DIR)
        except Exception:
            # If preload fails, regular request path will still retry and show user-facing error.
            pass

    threading.Thread(target=_load, daemon=True).start()


@app.route("/")
def index():
    ensure_model_preload()
    return render_template("index.html")


@app.route("/guest")
def guest():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    session["guest_mode"] = True
    flash("Гостьовий режим активовано.", "success")
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    ensure_model_preload()

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Усі поля обов'язкові.", "error")
            return redirect(url_for("register"))

        if not is_valid_email(email):
            flash("Введіть коректний email.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Пароль має містити мінімум 6 символів.", "error")
            return redirect(url_for("register"))

        existing = User.query.filter(or_(User.username == username, User.email == email)).first()
        if existing:
            flash("Користувач з таким логіном або email вже існує.", "error")
            return redirect(url_for("register"))

        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        flash("Реєстрація успішна. Тепер увійдіть у систему.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    ensure_model_preload()

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not is_valid_email(email):
            flash("Введіть коректний email.", "error")
            return redirect(url_for("login"))

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            session.pop("guest_mode", None)
            flash("Вхід виконано успішно.", "success")
            return redirect(url_for("dashboard"))

        flash("Невірний email або пароль.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("guest_mode", None)
    logout_user()
    flash("Ви вийшли з акаунта.", "success")
    return redirect(url_for("index"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    ensure_model_preload()

    if not current_user.is_authenticated and not session.get("guest_mode", False):
        flash("Спочатку увійдіть в акаунт або оберіть «Гість», щоб почати аналіз.", "error")
        return redirect(url_for("index"))

    result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        
        # URL Parsing feature
        if text.startswith("http://") or text.startswith("https://"):
            try:
                import requests
                from bs4 import BeautifulSoup
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(text, headers=headers, timeout=5)
                soup = BeautifulSoup(resp.text, "html.parser")
                paragraphs = soup.find_all("p")
                text = "\n".join(p.get_text() for p in paragraphs).strip()
                if not text:
                    flash("Не вдалося витягнути текст із сайту.", "error")
            except Exception as e:
                flash("Помилка парсингу посилання.", "error")
                text = ""

        if not text:
            flash("Введіть текст новини або коректне посилання.", "error")
        else:
            try:
                # Caching logic
                text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                cached = AnalysisCache.query.filter_by(text_hash=text_hash).first()
                if cached:
                    result = json.loads(cached.result_json)
                else:
                    result = ml_service.predict_selected_models(text, MODEL_BASE_DIR)
                    cache_entry = AnalysisCache(text_hash=text_hash, result_json=json.dumps(result, ensure_ascii=False))
                    db.session.add(cache_entry)
                    db.session.commit()

                if current_user.is_authenticated:
                    history_record = save_history_item(current_user.id, text, result)
                    result["history_id"] = history_record.id
                flash("Аналіз виконано.", "success")
            except Exception as exc:
                flash(f"Помилка аналізу: {exc}", "error")

    history_items = get_history_items(limit=5)
    return render_template(
        "dashboard.html",
        result=result,
        text=text,
        history_items=history_items,
        model_base_dir=MODEL_BASE_DIR,
    )


@app.route("/api/random_example/<example_type>")
@limiter.limit("5 per minute")
def random_example(example_type):
    import random
    import requests
    from bs4 import BeautifulSoup
    from flask import jsonify

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        if example_type == "fake":
            resp = requests.get("https://www.stopfake.org/uk/golovna/", headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a"):
                if a.has_attr("href") and ("/uk/fejk-" in a["href"] or "/uk/manipulyatsiya-" in a["href"] or "/uk/fotofejk-" in a["href"]):
                    links.append(a["href"])
            
            if not links:
                return jsonify({"error": "Не знайдено посилань на StopFake"}), 404
            
            chosen_url = random.choice(links)
            article_resp = requests.get(chosen_url, headers=headers, timeout=5)
            article_soup = BeautifulSoup(article_resp.text, "html.parser")
            content_div = article_soup.find("div", class_="entry-content")
            paragraphs = content_div.find_all("p") if content_div else article_soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs).strip()
            
            text = text.split("Більше спростувань")[0].strip()
            if len(text) > 3000:
                text = text[:3000] + "..."
            return jsonify({"text": text})
            
        elif example_type == "true":
            resp = requests.get("https://www.ukrinform.ua/rubric-ato", headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a"):
                if a.has_attr("href") and "/rubric-ato/" in a["href"] and a["href"].endswith(".html"):
                    url = a["href"]
                    if not url.startswith("http"):
                        url = "https://www.ukrinform.ua" + url
                    if url not in links:
                        links.append(url)
            
            if not links:
                return jsonify({"error": "Не знайдено посилань на Укрінформ"}), 404
                
            chosen_url = random.choice(links)
            article_resp = requests.get(chosen_url, headers=headers, timeout=5)
            article_soup = BeautifulSoup(article_resp.text, "html.parser")
            paragraphs = article_soup.find_all("p")
            text = "\n".join(p.get_text() for p in paragraphs).strip()
            if len(text) > 3000:
                text = text[:3000] + "..."
            return jsonify({"text": text})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
@login_required
def history():
    q = request.args.get("q", "").strip()
    query = CheckHistory.query.filter_by(user_id=current_user.id)
    if q:
        search = f"%{q}%"
        query = query.filter(db.or_(CheckHistory.input_text.ilike(search), CheckHistory.theme.ilike(search)))
    items = query.order_by(CheckHistory.created_at.desc()).all()
    return render_template("history.html", history_items=items, q=q)


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    CheckHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Історію перевірок очищено.", "success")
    next_url = request.referrer or url_for("dashboard")
    return redirect(next_url)


@app.route("/download-report", methods=["POST"])
def download_report():
    history_id = request.form.get("history_id", "").strip()

    if history_id:
        if not current_user.is_authenticated:
            abort(403)

        record = CheckHistory.query.filter_by(id=int(history_id), user_id=current_user.id).first()
        if record is None:
            abort(404)

        payload = {**record.analysis, "input_text": record.input_text}
        filename = f"news_report_{record.id}.pdf"
    else:
        analysis_json = request.form.get("analysis_json", "").strip()
        if not analysis_json:
            flash("Немає даних для PDF-звіту.", "error")
            return redirect(url_for("dashboard"))

        payload = json.loads(analysis_json)
        filename = "news_report_current.pdf"

    pdf_buffer = build_pdf_buffer(payload)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/feedback/<int:history_id>", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
def submit_feedback(history_id):
    is_correct = request.form.get("is_correct") == "1"
    existing = Feedback.query.filter_by(history_id=history_id, user_id=current_user.id).first()
    if existing:
        existing.is_correct = is_correct
    else:
        fb = Feedback(history_id=history_id, user_id=current_user.id, is_correct=is_correct)
        db.session.add(fb)
    db.session.commit()
    if request.headers.get("Fetch") == "true":
        return {"status": "ok"}
    flash("Дякуємо за відгук! Це допоможе покращити модель.", "success")
    return redirect(url_for("dashboard"))

@app.route("/admin")
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash("Доступ заборонено.", "error")
        return redirect(url_for("index"))
    
    users_count = User.query.count()
    checks_count = CheckHistory.query.count()
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).limit(50).all()
    
    # gather details for feedbacks
    feedback_data = []
    for fb in feedbacks:
        hist = CheckHistory.query.get(fb.history_id)
        user = User.query.get(fb.user_id)
        if hist and user:
            feedback_data.append({
                "id": fb.id,
                "email": user.email,
                "text": hist.input_text[:100] + "...",
                "is_correct": fb.is_correct,
                "date": fb.created_at
            })

    return render_template("admin.html", users=users_count, checks=checks_count, feedbacks=feedback_data)

@app.route("/admin/export_feedback")
@login_required
def export_feedback():
    if not current_user.is_admin:
        return "Access denied", 403
    import csv
    import io
    from flask import Response
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Feedback ID', 'User Email', 'Is Correct', 'Text', 'Fake Label', 'Prob True', 'Theme', 'Date'])
    
    feedbacks = Feedback.query.all()
    for fb in feedbacks:
        hist = CheckHistory.query.get(fb.history_id)
        user = User.query.get(fb.user_id)
        if hist and user:
            writer.writerow([fb.id, user.email, fb.is_correct, hist.input_text, hist.fake_label, hist.prob_true, hist.theme, fb.created_at])
            
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=feedback_export.csv"})

@app.cli.command("init-db")
def init_db_command():
    db.create_all()
    print("Database initialized")

import click
@app.cli.command("make-admin")
@click.argument("email")
def make_admin_command(email):
    user = User.query.filter_by(email=email).first()
    if user:
        user.is_admin = True
        db.session.commit()
        print(f"Success! {email} is now an admin.")
    else:
        print("User not found.")

with app.app_context():
    db.create_all()
    print("=== ROUTES ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint} -> {rule.rule}")
    print("==============")


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=debug_mode,
        use_reloader=False,
        load_dotenv=False,
    )
