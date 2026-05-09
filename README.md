# ⚖️ Algorithmic Fairness in Legal Education

> Analyzing and mitigating bias in bar passage prediction using the LSAC dataset (~20,000 law students)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![AIF360](https://img.shields.io/badge/IBM-AIF360-054ADA?style=for-the-badge&logo=ibm)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [References](#-references)

---

## 🔍 Overview

Machine learning models increasingly influence high-stakes decisions in education, finance, and healthcare. This project replicates and extends **Biswas and Rajan's** foundational study on ML fairness by:

- 🎯 Analyzing **race and gender bias** in bar exam passage predictions
- 🔧 Applying **two bias mitigation techniques**: Reweighing (pre-processing) and Reject Option Classification (post-processing)
- 📊 Evaluating **fairness-accuracy tradeoffs** across multiple metrics
- 🏛️ Focusing on the **legal education domain**, where biased predictions can shape the demographic composition of the entire legal profession

> **Why it matters:** If a predictive model used for law school admissions or academic interventions is biased, it can create a self-reinforcing cycle that disadvantages already marginalized groups.

---

## 💡 Key Findings

| Attribute | Bias Level | Best Mitigation | Bias Reduction | Accuracy Impact |
|-----------|-----------|-----------------|----------------|-----------------|
| Gender | Moderate | Reject Option Classification (ROC) | **96%** reduction in SPD | +0.55% ✅ |
| Race | Severe | Reject Option Classification (ROC) | **54%** reduction (with overcorrection) | -0.18% ✅ |

### Highlights
- 🔴 **Race bias was far more severe** — non-white students had a **51.38% lower** probability of favorable outcomes at baseline
- 🟡 **Gender bias was moderate** — males had an **11.87% higher** favorable outcome rate
- ✅ **ROC outperformed Reweighing** for both protected attributes
- ⚠️ **Race bias proved resistant** to complete mitigation due to multiple interacting feature pathways
- 🎯 **Accuracy-fairness tradeoff was minimal** in this domain (vs. 7.5% penalty in original study)

---

## 📂 Dataset

**Source:** Law School Admissions Council (LSAC) National Longitudinal Bar Passage Study

| Property | Details |
|----------|---------|
| Size | ~20,000 law students |
| Year | 1991 cohort |
| Task | Binary classification (bar exam pass/fail) |
| Class Balance | 81.7% pass rate (imbalanced) |

### Demographic Distribution
Gender  →  56.1% Male   |  43.9% Female
Race    →  82.4% White  |  17.6% Non-White

### Raw Outcome Disparities
White students     →  84.2% pass rate
Non-white students →  68.5% pass rate
Male students      →  83.1% pass rate
Female students    →  79.9% pass rate

### Features Used

| Feature | Importance | Notes |
|---------|-----------|-------|
| Law School GPA | 0.43 | Strongest predictor |
| LSAT Score | 0.21 | Also correlated with race |
| Undergraduate GPA | 0.15 | — |
| Law School Tier | — | Structural disadvantage for non-white students |
| Extracurricular Participation | — | Binary indicator |
| Prior Work Experience | — | Years before law school |

---

## 🔬 Methodology

### Research Pipeline
Raw LSAC Data
│
▼
Data Preprocessing
(encoding, missing values, protected attribute selection)
│
▼
Train / Test Split  (70% / 30%)
│
▼
Logistic Regression Model
(Accuracy: 80.8%, AUC-ROC: 0.865)
│
▼
Fairness Evaluation (4 metrics)
│
├──► Reweighing (Pre-processing)
│
└──► Reject Option Classification (Post-processing)
│
▼
Fairness + Accuracy Report

### Fairness Metrics Used

| Metric | Ideal Value | What it Measures |
|--------|------------|-----------------|
| Statistical Parity Difference (SPD) | 0 | Difference in favorable outcome rates |
| Disparate Impact (DI) | 1 | Ratio of favorable outcomes between groups |
| Equal Opportunity Difference (EOD) | 0 | Difference in true positive rates |
| Average Odds Difference (AOD) | 0 | Average of TPR and FPR differences |

### Mitigation Techniques

**1. Reweighing (Pre-processing)**
- Assigns different weights to training instances based on protected attribute + outcome combinations
- Goal: reduce correlation between protected attributes and outcomes
- Best weight range: `0.6–0.8` for gender, `0.8–0.9` for race

**2. Reject Option Classification — ROC (Post-processing)**
- Modifies predictions near the decision boundary based on protected attribute
- Flips predictions for borderline cases (~3.2% of predictions for gender)
- Best threshold: `0.7–0.8` for gender, `0.5–0.6` for race

---

## 📊 Results

### Gender Fairness Metrics

| Metric | Before Mitigation | After Reweighing | After ROC |
|--------|:-----------------:|:----------------:|:---------:|
| SPD | 0.0577 | 0.0608 | **-0.0023** ✅ |
| DI | 1.1187 | 1.1260 | **0.9959** ✅ |
| EOD | 0.0413 | 0.0474 | **-0.0258** ✅ |
| AOD | 0.0501 | 0.0531 | **-0.0096** ✅ |
| Accuracy | 80.80% | 80.62% | **81.35%** ✅ |

### Race Fairness Metrics

| Metric | Before Mitigation | After Reweighing | After ROC |
|--------|:-----------------:|:----------------:|:---------:|
| SPD | -0.5138 | -0.5120 | **0.2362** ⚠️ |
| DI | 0.0000 | 0.0000 | **1.4597** ⚠️ |
| EOD | -0.8070 | -0.8035 | **0.1930** ⚠️ |
| AOD | -0.4985 | -0.4967 | **0.3349** ⚠️ |
| Accuracy | 80.80% | 80.62% | **80.62%** ✅ |

> ⚠️ ROC overcorrected for race bias due to extreme initial disparity — reversing bias direction rather than eliminating it.

### Accuracy–Fairness Tradeoff
High Fairness │  ●  After Reweighing
│
│        ●  Baseline
│
Low Fairness  │                    ●  After ROC
└─────────────────────────────────
Low Accuracy        High Accuracy
> ROC achieves the best accuracy with the lowest fairness score (closest to 0 = most fair).

---

## 🗂️ Project Structure
algorithmic-fairness-law-school/
│
├── 📁 data/
│   └── lawschool.csv                          # LSAC dataset
│
├── 📁 notebooks/
│   └── Program.ipynb                          # Full analysis pipeline
│
├── 📁 report/
│   ├── main.tex                               # LaTeX source
│   └── FDA_Project_Report.pdf                # Final paper
│
├── 📁 results/
│   ├── fairness_analysis_results.txt          # Raw metrics output
│   ├── bias_comparison_png.jpeg               # Cross-dataset bias chart
│   ├── gender_mitigation_png.png             # Gender fairness comparison
│   ├── race_mitigation_png.png               # Race fairness comparison
│   ├── research_methodology_diagram_png.jpeg  # Pipeline diagram
│   └── tradeoff_plot.png                     # Accuracy-fairness tradeoff
│
└── README.md

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AmanKumar-23/algorithmic-fairness-law-school.git
cd algorithmic-fairness-law-school
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn aif360 jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook notebooks/Program.ipynb
```

### 4. Run all cells in order
The notebook will:
- Load and preprocess the LSAC dataset
- Train the baseline logistic regression model
- Compute all fairness metrics
- Apply both mitigation techniques
- Generate all plots saved in `/results`

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Jupyter Notebook | Analysis environment |
| scikit-learn | Logistic regression model |
| IBM AIF360 | Fairness metrics + mitigation |
| pandas / numpy | Data processing |
| matplotlib / seaborn | Visualizations |
| LaTeX | Research paper |

---

## 👥 Author

**Department of Computer Science and Engineering**
**Faculty of Technology, University of Delhi**

| Name | GitHub |
|------|--------|
| Aman Kumar | [@AmanKumar-23](https://github.com/AmanKumar-23) |

---

## 📚 References

1. Biswas & Rajan — *Do the machine learning models on a crowd sourced platform exhibit bias?* (ESEC/FSE 2020)
2. Kamiran & Calders — *Data preprocessing techniques for classification without discrimination* (2012)
3. Kamiran et al. — *Decision theory for discrimination-aware classification* (IEEE ICDM 2012)
4. Bellamy et al. — *AI Fairness 360: An extensible toolkit* (IBM Journal, 2019)
5. Hardt, Price & Srebro — *Equality of opportunity in supervised learning* (NeurIPS 2016)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

<div align="center">

**⭐ If you found this project useful, please consider giving it a star!**

*Built with ❤️ for fair and accountable machine learning*

</div>
