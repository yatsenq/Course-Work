# Fake News Detector for Ukrainian News 🇺🇦🤖

An intelligent web system for automatic fake news detection and thematic classification of Ukrainian news using Machine Learning and Deep Learning (mBERT).

## 🌟 Features
- **Binary Classification:** Detects if a news article is `TRUE` or `FAKE` with high confidence.
- **Topic Classification:** Automatically categorizes news into 4 themes: Politics, Sport, Business, and Technology.
- **Hybrid AI Model:** Uses an ensemble of **mBERT** (Multilingual BERT) and classic **Logistic Regression** with TF-IDF.
- **Automated Evidence:** Highlights specific manipulation markers (keywords) in the text.
- **Reporting:** Generates detailed PDF reports for each analysis.
- **User Dashboard:** Full history of checks for authorized users.
- **Admin Panel:** Real-time stats and user management.

## 🛠 Tech Stack
- **Backend:** Python, Flask, Flask-Login, SQLAlchemy
- **AI/ML:** PyTorch, Transformers (Hugging Face), Scikit-learn
- **NLP:** PyMorphy3 (for Ukrainian morphology)
- **Database:** PostgreSQL / SQLite
- **Deployment:** Docker, Hugging Face Spaces

## 🏗 System Architecture
The system uses a **distributed deployment model** to optimize resources:
1. **Model Repository:** Stores heavy BERT weights and vectorized models.
2. **App Space:** Runs the Flask web application in a Docker container.
3. **ML Service:** Dynamically pulls weights on startup and performs inference using an ensemble of 4 models.

## 🚀 Installation & Local Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yatsenq/coursework-fake-news-detector.git
   cd coursework-fake-news-detector/Deployment/flask_app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file with `SECRET_KEY` and `DATABASE_URL`.
4. Run the app:
   ```bash
   python app.py
   ```

## 📊 Performance
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| mBERT (Head C) | **92.9%** | **0.929** |
| Logistic Regression | 84.1% | 0.837 |
| SVM (Topic) | 96.7% | 0.967 |

## 📝 Author
**Yatsenko V. T.**  
Student at Ivan Franko National University of Lviv  
Coursework 2026

---
*Disclaimer: This tool is an AI-powered assistant. Please double-check critical information via official sources.*
