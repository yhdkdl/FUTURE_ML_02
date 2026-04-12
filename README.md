# 🎫 Support Ticket Classifier

An end-to-end ML system that automatically classifies IT support tickets
by **category** and **priority** using NLP and scikit-learn.

Built on 2,229 real human-classified IT support tickets from a Brazilian
IT company (Zenodo dataset).

---

## 📊 Results

| Task | Model | Accuracy | F1 Score |
|---|---|---|---|
| Category Classification (7 classes) | LinearSVC | 78.4% | 0.787 |
| Priority Prediction (4 levels) | LinearSVC | 86.8% | 0.868 |

### Per-Class Category Performance

| Category | F1 Score |
|---|---|
| EOL | 1.00 |
| Fileservice | 0.94 |
| Support general | 0.80 |
| Software | 0.67 |
| O365 | 0.68 |
| Computer-Services | 0.65 |
| Active Directory | 0.49 |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/support-ticket-classifier
cd support-ticket-classifier
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python -c "
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
"
```

### 5. Download dataset
Download from [Zenodo](https://zenodo.org/records/7648117):
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

Place all 4 files in `data/raw/`

### 6. Train the models
Run notebooks in order:
```
notebooks/02_preprocessing.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_category_model.ipynb
notebooks/05_priority_model.ipynb
```

---

## 💻 CLI Usage

### Classify a single ticket
```bash
python main.py --ticket "Cannot access shared folder on network drive"
```

Output:
```
=======================================================
  Ticket Classification Result
=======================================================
  Input:    Cannot access shared folder on network drive
  Category: Fileservice
  Priority: Critical (Level 3/3)
  Valid:    True
=======================================================
```

### Classify a CSV file of tickets
```bash
python main.py --file data/raw/X_test.csv --column text
```

### Run live demo
```bash
python main.py --demo
```

### Show system info
```bash
python main.py --info
```

---

## 🏗️ Project Structure

```
support-ticket-classifier/
├── data/
│   ├── raw/              # Original Zenodo CSV files
│   └── processed/        # Cleaned train/test CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_category_model.ipynb
│   ├── 05_priority_model.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_inference_pipeline.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── priority_labeler.py
│   ├── features/
│   │   └── vectorizer.py
│   ├── models/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── reporter.py
│   ├── pipeline/
│   │   └── predictor.py
│   ├── cli/
│   │   └── commands.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── outputs/
│   ├── models/           # Saved .pkl model files
│   ├── figures/          # Charts and confusion matrices
│   └── reports/          # Evaluation reports
├── tests/
│   ├── test_preprocessor.py
│   ├── test_predictor.py
│   └── test_cli.py
├── main.py
└── requirements.txt
```

---

## 🧠 How It Works

### 1. Text Preprocessing
Raw ticket text is cleaned through a pipeline:
- Lowercase conversion
- Placeholder removal (`{product_purchased}`, `[NAME]`)
- Boilerplate removal (template phrases)
- Punctuation and number removal
- Stopword removal
- Lemmatization

### 2. Feature Extraction
TF-IDF vectorization with:
- 1,330 features (vocabulary size)
- Unigrams and bigrams (`ngram_range=(1,2)`)
- `max_df=0.70` — removes words in 70%+ of tickets
- `sublinear_tf=True` — log scaling

### 3. Classification
Two independent LinearSVC models:
- **Category model** — classifies into 7 IT support categories
- **Priority model** — predicts urgency level (Low/Medium/High/Critical)

### 4. Priority Assignment Logic

| Category | Priority | Reason |
|---|---|---|
| EOL | Low | Planned decommission work |
| Computer-Services | Medium | Standard hardware requests |
| O365 | Medium | Software productivity issues |
| Software | Medium | Application issues |
| Active Directory | High | Blocks user access |
| Support general | High | Escalated issues |
| Fileservice | Critical | Blocks team file access |

---



## 📈 Key Findings

1. **LinearSVC consistently outperforms** Logistic Regression and
   Random Forest on this text classification task

2. **Multilingual tickets** (German, Portuguese, Spanish) are handled
   naturally — foreign words cluster in specific categories

3. **Class imbalance** (EOL: 45 vs Fileservice: 546 tickets) handled
   with `class_weight='balanced'`

4. **EOL achieves perfect F1=1.00** due to highly distinctive vocabulary
   (`nexthink`, `end of life`, `decommission`)

5. **Active Directory scores lowest (F1=0.49)** because AD ticket language
   overlaps significantly with Support general tickets

6. **433 tickets/second** throughput — production-grade inference speed

---

## 📁 Dataset

**Classification of IT Support Tickets**
Zenodo DOI: [10.5281/zenodo.7648117](https://doi.org/10.5281/zenodo.7648117)

2,229 support tickets manually classified by IT professionals.
Real tickets from a Brazilian IT support company (2020).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| scikit-learn | ML models and TF-IDF |
| NLTK | Text preprocessing |
| pandas | Data manipulation |
| matplotlib / seaborn | Visualisation |
| joblib | Model serialisation |

---

## 👤 Author
Built as part of a structured ML learning sprint series.