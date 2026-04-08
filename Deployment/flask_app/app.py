from __future__ import annotations

import io
import json
import os
import secrets
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import or_
from werkzeug.security import check_password_hash, generate_password_hash

from ml_service import predict_news_bundle


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

MODEL_BASE_DIR = os.getenv("MODEL_BASE_DIR", str(Path(__file__).resolve().parents[2]))

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
    password_hash = db.Column(db.String(255), nullable=False)
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


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def save_history_item(user_id: int, text: str, result: dict) -> None:
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


def get_history_items(limit: int = 12):
    if not current_user.is_authenticated:
        return []

    return (
        CheckHistory.query.filter_by(user_id=current_user.id)
        .order_by(CheckHistory.created_at.desc())
        .limit(limit)
        .all()
    )


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
    story.append(Table(fake_rows, colWidths=[55 * mm, 90 * mm]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Тематична класифікація", section_style))
    topic_rows = [
        ["Категорія", payload.get("theme", "—")],
        ["Впевненість", f"{payload.get('theme_confidence', 0) * 100:.2f}%"],
    ]
    story.append(Table(topic_rows, colWidths=[55 * mm, 90 * mm]))
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
    return {
        "examples": EXAMPLE_NEWS,
        "guest_mode": session.get("guest_mode", False) and not current_user.is_authenticated,
    }


@app.route("/")
def index():
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
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

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
    result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if not text:
            flash("Введіть текст новини.", "error")
        else:
            try:
                result = predict_news_bundle(text, MODEL_BASE_DIR)
                result["input_text"] = text
                if current_user.is_authenticated:
                    save_history_item(current_user.id, text, result)
                flash("Аналіз виконано.", "success")
            except Exception as exc:
                flash(f"Помилка аналізу: {exc}", "error")

    history_items = get_history_items()
    return render_template(
        "dashboard.html",
        result=result,
        text=text,
        history_items=history_items,
        model_base_dir=MODEL_BASE_DIR,
    )


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    CheckHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Історію перевірок очищено.", "success")
    return redirect(url_for("dashboard"))


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


@app.cli.command("init-db")
def init_db_command():
    db.create_all()
    print("Database initialized")


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
