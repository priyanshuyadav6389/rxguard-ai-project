# RxGuard AI v2 💊
### AI-Powered Drug Interaction & Prescription Safety Checker

## Features
- 🔍 Drug interaction detection (severe / moderate / mild)
- 💊 Brand name → generic drug resolution (200+ brand names)
- 🤰 Pregnancy safety checker
- 👴 Age-based Beers Criteria warnings
- ⚠️ Allergy class detection
- 🫀 Condition-based contraindications (kidney, liver, heart, diabetes)
- 🎯 AI Risk Score (0–100)
- 🕸️ D3.js interactive interaction graph
- 🗺️ Risk heatmap matrix
- 🤖 ML model comparison dashboard (Random Forest, SVM, Logistic Regression, ANN)
- 📄 PDF / Print / Copy report export
- 📊 Training plots (confusion matrix, feature importance, ANN curves)

## Datasets Used
- **DrugBank Open Data** — 100+ drugs, categories, mechanisms
- **TWOSIDES (Tatonetti Lab)** — drug pair side effect interactions
- **FDA FAERS** — adverse event reporting data

## Tech Stack
- Backend: Python, Flask, Pandas, NumPy, Scikit-learn, Matplotlib
- ML: Random Forest, Logistic Regression, SVM, Gradient Boosting, Keras ANN
- Frontend: Vanilla HTML/CSS/JS + D3.js
- Deployment: Render (free tier)

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/rxguard-ai
cd rxguard-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
python download_datasets.py
python train_model.py
python app.py
```
Open http://localhost:5000

## Syllabus Coverage
| Module | Usage |
|--------|-------|
| Python + NumPy/Pandas | Drug DB, data processing |
| Search (BFS/A*) | Interaction graph traversal |
| ML (SVM, RF, LR) | Severity classification |
| Deep Learning (Keras) | ANN prediction + training curves |
