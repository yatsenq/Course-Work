# Експериментальні Jupyter Notebooks

## Огляд створених notebooks

Всього створено **8 комплексних Jupyter notebooks** для експериментів з детекцією фейкових новин та класифікацією тем.

---

## 📊 Структура експериментів

### Fake News Detection (Детекція фейкових новин)

#### 02_bert_fake_base.ipynb

**Базовий експеримент з трьома архітектурами BERT**

- 🎯 **Мета**: Порівняння 3 різних класифікаційних голов
- 📦 **Датасет**: Повний збалансований FINAL_DATASET.csv (70/15/15)
- 🏗️ **Архітектури**:
  - HeadA: `[CLS]→Dropout(0.1)→Linear(768,2)`
  - HeadB: `[CLS]→Linear(768,256)→BatchNorm→ReLU→Dropout(0.1)→Linear(256,2)`
  - HeadC: `AttentionPooling→LayerNorm→Dropout(0.1)→Linear(768,2)`
- 📈 **Візуалізації**: Confusion matrix, ROC, learning curves, epoch delta для кожної голови + комбіновані графіки
- 💾 **Зберігає**: 3 моделі (.pt), tokenizer, конфігурації, метрики

#### 03_bert_fake_expA.ipynb

**Експеримент з дисбалансом класів**

- 🎯 **Мета**: Дослідити вплив дисбалансу (100% TRUE + 50% FAKE)
- 📦 **Датасет**: Незбалансований (random_state=42 для sampling)
- 🏗️ **Архітектури**: Ті ж 3 голови з ідентичними гіперпараметрами
- 📈 **Візуалізації**: Всі з 02 + порівняльні графіки Base vs ExpA + деградаційні діаграми
- 💡 **Ключова метрика**: Зниження recall для FAKE класу

---

### Topic Classification (Класифікація тем)

#### 04_topic_svm_base.ipynb

**Baseline SVM з TF-IDF**

- 🎯 **Мета**: Класифікація за 5 темами (політика, спорт, новини, технології, бізнес)
- 📦 **Датасет**: ua_news_train_balanced.csv (80/20 split)
- 🔧 **Підхід**: LinearSVC + CalibratedClassifierCV + TF-IDF(1,2)-grams
- 🧹 **Preprocessing**: lowercase, stopwords, PyMorphy3 лематизація
- 📈 **Візуалізації**: Confusion matrix 5x5, per-class F1, топ-15 слів на клас
- 💾 **Зберігає**: model.pkl, vectorizer.pkl

#### 05_topic_svm_expB.ipynb

**SVM без категорії "новини"**

- 🎯 **Мета**: Перевірити вплив видалення неспецифічної категорії
- 📦 **Датасет**: Відфільтрований (4 класи)
- 🔧 **Підхід**: Ідентичні гіперпараметри до 04
- 📈 **Візуалізації**: Confusion matrix 4x4, порівняння Base vs ExpB
- 💡 **Очікування**: Покращення через більшу специфічність класів

#### 06_bert_topic_base.ipynb

**BERT для класифікації тем (baseline)**

- 🎯 **Мета**: BERT HeadA для 5 класів
- 📦 **Датасет**: Повний (70/15/15 split)
- 🔧 **Параметри**: max_length=128 (коротше для тем), epochs=3, lr=2e-5
- 📈 **Візуалізації**: Confusion matrix 5x5, learning curves, epoch delta, per-class F1
- 📊 **Додатково**: Порівняння SVM Base vs BERT Base
- 💾 **Зберігає**: bert_topic_base.pt, label_encoder.pkl

#### 07_bert_topic_expB.ipynb

**BERT без категорії "новини"**

- 🎯 **Мета**: BERT HeadA для 4 класів
- 📦 **Датасет**: Відфільтрований
- 🔧 **Параметри**: Linear(768, 4) замість Linear(768, 5)
- 📈 **Візуалізації**: Confusion matrix 4x4, learning curves, epoch delta
- 📊 **Додатково**: Порівняння всіх 4 topic моделей (SVM base, SVM expB, BERT base, BERT expB)

---

### Combined & Final Experiments

#### 08_experiment_C_combined.ipynb

**Worst-case комбінований сценарій**

- 🎯 **Мета**: Тестування в найгіршому випадку
- 📦 **Умови**:
  - Fake: 50% FAKE (дисбаланс)
  - Topic: Без "новини" (4 класи)
- 🏗️ **Моделі**:
  - Fake: Найкраща голова з 02 (HeadA default)
  - Topic: LinearSVC
- 📈 **Візуалізації**: Порівняння з baseline для обох задач
- 💡 **Ключ**: Оцінка деградації метрик у складних умовах

#### 09_final_comparison.ipynb

**Фінальне зведення всіх результатів**

- 🎯 **Мета**: Комплексне порівняння всіх експериментів
- 📊 **Таблиця 1**: Всі fake моделі (accuracy, precision, recall, F1, ROC-AUC) з підсвічуванням максимумів
- 📊 **Таблиця 2**: Всі topic моделі (accuracy, macro-F1, weighted-F1, per-class F1)
- 📈 **Візуалізації**: Групові барплоти, порівняння Base vs Experimental
- 💾 **Зберігає**: Зведені CSV таблиці, фінальний JSON звіт
- 📝 **Висновки**: Детальний аналіз та рекомендації

---

## 🔧 Технічні деталі (спільні для всіх)

### Обов'язкові вимоги

✅ **SEED=42** всюди (random, numpy, torch.manual_seed, torch.cuda.manual_seed_all)  
✅ **Device** = cuda if available else cpu  
✅ **tqdm** для всіх циклів  
✅ **Формат виводу**: `"Epoch N/3 | Train Loss: X | Val Loss: Y | Delta: Z"`  
✅ **plt.show()** після inline plots, **plt.close()** після збереження  
✅ **assert os.path.exists(dataset_path)** на початку  
✅ Імпорт з `../utils/experiment_utils`  
✅ Збереження: models→`../models/`, plots→`../plots/`, metrics→`../results/`

### BERT Training Loop (02, 03, 06, 07, 08)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=int(0.1*total_steps),
    num_training_steps=total_steps)
```

### AttentionPooling Architecture (HeadC)

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)
```

---

## 📂 Структура файлів

```
experiments/
├── 02_bert_fake_base.ipynb          # BERT 3 heads на full датасеті
├── 03_bert_fake_expA.ipynb          # BERT на незбалансованому (50% FAKE)
├── 04_topic_svm_base.ipynb          # SVM для 5 класів
├── 05_topic_svm_expB.ipynb          # SVM для 4 класів (без "новини")
├── 06_bert_topic_base.ipynb         # BERT для 5 класів
├── 07_bert_topic_expB.ipynb         # BERT для 4 класів
├── 08_experiment_C_combined.ipynb   # Worst-case комбінований
├── 09_final_comparison.ipynb        # Фінальне порівняння
│
├── models/                           # Збережені моделі
│   ├── bert_fake_headA.pt
│   ├── bert_fake_headB.pt
│   ├── bert_fake_headC.pt
│   ├── bert_fake_expA_headA.pt
│   ├── ...
│   ├── topic_svm_base_model.pkl
│   └── ...
│
├── plots/                            # Збережені графіки (PNG 300 dpi)
│   ├── 02_bert_fake_heada_results.png
│   ├── 02_bert_fake_combined_val_loss.png
│   ├── ...
│   ├── 09_fake_all_models_comparison.png
│   └── 09_base_vs_experimental_comparison.png
│
└── results/                          # JSON з метриками
    ├── 02_bert_fake_headA_results.json
    ├── ...
    ├── 09_fake_detection_summary.csv
    ├── 09_topic_classification_summary.csv
    └── 09_final_summary.json
```

---

## 🚀 Послідовність запуску

### Рекомендований порядок:

1. **Fake Detection Baseline**:

   ```
   02_bert_fake_base.ipynb
   ```

2. **Fake Detection Experimental**:

   ```
   03_bert_fake_expA.ipynb
   ```

3. **Topic Classification Baselines**:

   ```
   04_topic_svm_base.ipynb
   06_bert_topic_base.ipynb
   ```

4. **Topic Classification Experimental**:

   ```
   05_topic_svm_expB.ipynb
   07_bert_topic_expB.ipynb
   ```

5. **Combined Worst-Case**:

   ```
   08_experiment_C_combined.ipynb
   ```

6. **Final Analysis**:
   ```
   09_final_comparison.ipynb
   ```

---

## 📊 Ключові метрики

### Fake Detection

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Topic Classification

- Accuracy
- Macro F1-Score
- Weighted F1-Score
- Per-class F1-Scores

---

## 🎨 Візуалізації

Кожен notebook створює:

- **Confusion matrices** (seaborn heatmaps)
- **Learning curves** (train/val loss)
- **Epoch delta charts** (overfitting indicators)
- **ROC curves** (для binary classification)
- **Grouped bar plots** (метрики comparison)
- **Per-class F1 bars** (для multiclass)
- **Top words visualization** (для SVM)

Всі графіки зберігаються у високій роздільності (300 dpi) для публікації.

---

## 💡 Висновки та інсайти

Детальний аналіз результатів знаходиться в **09_final_comparison.ipynb**, включаючи:

1. **Вплив архітектури** (HeadA vs HeadB vs HeadC)
2. **Вплив дисбалансу класів** (Base vs ExpA)
3. **SVM vs BERT** порівняння
4. **Вплив видалення категорії** (5 vs 4 класи)
5. **Worst-case performance** (ExpC)
6. **Рекомендації для продакшену**

---

## 📝 Українська локалізація

Всі notebooks містять:

- ✅ Українські коментарі
- ✅ Українські markdown секції
- ✅ Українські labels на графіках
- ✅ Українські назви класів (політика, спорт, новини, технології, бізнес)

---

## ⚙️ Залежності

```python
# Core ML
torch, transformers, sklearn

# Data processing
pandas, numpy

# Visualization
matplotlib, seaborn

# Ukrainian NLP
pymorphy3

# Utils
tqdm, warnings
```

---

**Створено**: 8 comprehensive Jupyter notebooks  
**Мова коментарів**: Українська  
**Стиль коду**: Production-ready з proper error handling  
**Документація**: Extensive markdown cells

🎓 **Готово для курсової роботи!**
