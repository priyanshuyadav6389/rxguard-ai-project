# 💊 RxGuard AI v2
### 🩺 AI-Powered Drug Interaction & Prescription Safety Checker

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras"/>
  <img src="https://img.shields.io/badge/D3.js-Visualization-F9A03C?style=for-the-badge&logo=d3.js&logoColor=white"/>
  <img src="https://img.shields.io/badge/Open%20Source-Contributions%20Welcome-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <strong>RxGuard AI helps identify dangerous drug interactions, predicts prescription risks using Machine Learning, and provides intelligent safety recommendations for patients and healthcare professionals.</strong>
</p>

---



## 📂 GitHub Repository

🔗 **Repository:**  
https://github.com/priyanshuyadav6389/rxguard-ai-project

---

# 📖 About the Project

**RxGuard AI** is an AI-powered healthcare application that analyzes prescriptions and detects potentially dangerous drug interactions using Machine Learning and medical datasets.

The platform assists doctors, pharmacists, and patients by identifying medication conflicts, allergies, pregnancy risks, age-related concerns, and disease-specific contraindications before medications are prescribed.

It combines **Artificial Intelligence**, **Machine Learning**, and **interactive data visualization** to improve medication safety and reduce the risk of adverse drug reactions.

---

# ✨ Key Features

### 💊 Drug Safety Analysis

- 🔍 Drug Interaction Detection
- 💊 Brand Name → Generic Drug Conversion
- ⚠️ Allergy Detection
- 🤰 Pregnancy Safety Checker
- 👴 Age-Based (Beers Criteria) Warnings
- ❤️ Condition-Based Contraindications
  - Kidney Disease
  - Liver Disease
  - Heart Disease
  - Diabetes

---

### 🤖 Artificial Intelligence

- 🎯 AI Risk Score (0–100)
- 🧠 Machine Learning Prediction
- 📊 Prescription Risk Classification
- 📈 Model Performance Dashboard

---

### 📊 Interactive Visualizations

- 🕸️ D3.js Drug Interaction Network
- 🗺️ Risk Heatmap Matrix
- 📉 Confusion Matrix
- 📊 Feature Importance Graph
- 📈 ANN Training Curves
- 📋 Model Comparison Dashboard

---

### 📄 Reporting

- 📑 Generate PDF Reports
- 🖨️ Print Reports
- 📋 Copy Results
- 📊 Detailed Risk Summary

---

# 🚀 Tech Stack

## 🖥️ Backend

- 🐍 Python
- ⚡ Flask
- 📊 Pandas
- 🔢 NumPy

---

## 🤖 Machine Learning

- 🌲 Random Forest
- 📉 Logistic Regression
- 📈 Support Vector Machine (SVM)
- 🌳 Gradient Boosting
- 🧠 Artificial Neural Network (Keras)

---

## 🎨 Frontend

- HTML5
- CSS3
- JavaScript
- D3.js

---

## 📦 Deployment

- Render

---

# 📂 Project Structure

```text
rxguard-ai/
│
├── app.py
├── train_model.py
├── prepare_data.py
├── download_datasets.py
├── requirements.txt
│
├── datasets/
│
├── models/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│
├── reports/
│
└── README.md
```

---

# 🗂️ Datasets Used

| Dataset | Purpose |
|----------|----------|
| 💊 DrugBank Open Data | Drug database, mechanisms, categories |
| ⚠️ TWOSIDES Dataset | Drug-drug interaction analysis |
| 🏥 FDA FAERS | Adverse drug event reporting |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/priyanshuyadav6389/rxguard-ai-project.git
```

## Move into Project

```bash
cd rxguard-ai
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Prepare Dataset

```bash
python prepare_data.py
```

---

## Download Required Datasets

```bash
python download_datasets.py
```

---

## Train Machine Learning Models

```bash
python train_model.py
```

---

## Run the Application

```bash
python app.py
```

Visit:

```
http://localhost:5000
```

---

# 🧠 Machine Learning Models

The project compares multiple Machine Learning algorithms to determine the most accurate prediction model.

| Model | Purpose |
|--------|----------|
| 🌲 Random Forest | Drug Risk Prediction |
| 📉 Logistic Regression | Binary Classification |
| 📈 Support Vector Machine | Severity Detection |
| 🌳 Gradient Boosting | Improved Accuracy |
| 🧠 Artificial Neural Network | Deep Learning Prediction |

---

# 📊 Visual Dashboard

The application includes interactive dashboards such as:

- 📊 Risk Score Meter
- 🕸️ Drug Interaction Network
- 🔥 Risk Heatmap
- 📈 Model Comparison Graph
- 📉 Confusion Matrix
- 📊 Feature Importance
- 📈 ANN Learning Curves

---

# 🎯 Project Objectives

- 💊 Improve prescription safety
- 🤖 Apply AI in healthcare
- ⚠️ Detect harmful drug interactions
- 📈 Compare Machine Learning models
- 🩺 Assist doctors and pharmacists
- 📊 Visualize medical data effectively

---

# 📚 Academic Concepts Covered

| Subject | Implementation |
|----------|----------------|
| 🐍 Python Programming | Backend Development |
| 📊 Pandas & NumPy | Data Processing |
| 🔍 Graph Search Algorithms | Drug Interaction Network |
| 🤖 Machine Learning | Risk Prediction |
| 🧠 Deep Learning | ANN Model |
| 🌐 Flask | Web Framework |
| 📈 Data Visualization | D3.js Charts |

---

# 🔮 Future Enhancements

- 🤖 Large Language Model (LLM) Medical Assistant
- 🗣️ Voice Prescription Analysis
- 📷 OCR-Based Prescription Scanner
- ☁️ Cloud Database Integration
- 👨‍⚕️ Doctor Login Portal
- 👤 Patient Dashboard
- 💬 AI Medical Chatbot
- 📱 Mobile Application
- 🌙 Dark Mode
- 🔔 Medication Reminder Notifications

---

# 🤝 Contributing

Contributions are welcome!

### Steps

1. Fork this repository

2. Create a feature branch

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes

```bash
git commit -m "Add New Feature"
```

4. Push your branch

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request 🚀

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

Your support encourages future improvements and helps others discover the project.

---

# 👨‍💻 Author

## Priyanshu Yadav

🐙 **GitHub**

https://github.com/priyanshuyadav6389

📂 **Project Repository**

https://github.com/priyanshuyadav6389/rxguard-ai-project

---

# 📄 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute it for educational and research purposes.

---

<div align="center">

# 💙 Making Medication Safer with Artificial Intelligence

### 🩺 AI • Healthcare • Machine Learning • Patient Safety

⭐ **If you like this project, don't forget to Star the Repository!** ⭐

Made with ❤️ by **Priyanshu Yadav**

</div>
