# Використовуємо образ Python 3.11 з мінімальним розміром
FROM python:3.11-slim

# Встановлюємо системні залежності для роботи бази даних та компіляції деяких бібліотек
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо список залежностей та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код додатка в образ
COPY . .

# Налаштування середовища
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Hugging Face Spaces очікує додаток на порту 7860
EXPOSE 7860

# Запускаємо сервер Gunicorn
# --timeout 120 потрібен, бо BERT моделі можуть довго завантажуватися при старті
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "app:app"]
