# Flask Website (Auth + ML)

## Що реалізовано

- Реєстрація і вхід користувачів
- SQLite база даних для акаунтів
- Захищений кабінет для аналізу новин
- Інтеграція з BERT + SVM моделями
- Docker + docker-compose запуск

## Локальний запуск

1. Активуй віртуальне середовище:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
2. Встанови залежності:
   - `pip install -r Deployment/flask_app/requirements.txt`
3. Запуск:
   - `python Deployment/flask_app/app.py`
4. Відкрий:
   - `http://127.0.0.1:5000`

## Docker запуск

1. Із кореня проєкту:
   - `docker compose up --build`
2. Відкрий:
   - `http://127.0.0.1:5000`

## Важливо

- Моделі читаються з шляху `MODEL_BASE_DIR/Models/...`
- У docker-compose вже задано `MODEL_BASE_DIR=/workspace`.
