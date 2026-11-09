# 🩺 Drug Knowledge Base & Expert System

An AI-powered clinical decision-support prototype for **drug-drug interaction (DDI) severity prediction** and drug safety lookup. It combines a **knowledge graph** (NetworkX), **TF-IDF text features**, and ensemble classifiers, and is served through a Streamlit dashboard with a PDF prescription generator.

## 📊 Results

Multi-class severity prediction (**Major / Moderate / Minor**) on a stratified 80/20 split: **104,337 training pairs** and **26,085 test pairs**.

| Model | Train Acc. | Test Acc. | Weighted F1 |
|---|---|---|---|
| **Random Forest (champion)** | 0.8205 | **0.7468** | **0.7605** |
| XGBoost | 0.7479 | 0.7169 | 0.7417 |
| LightGBM | 0.7033 | 0.6843 | 0.7150 |
| CatBoost | 0.5278 | 0.5205 | 0.5639 |

**Champion model, per class (Random Forest):**

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Major | 0.52 | 0.72 | 0.60 | 5,383 |
| Minor | 0.40 | 0.64 | 0.49 | 1,367 |
| Moderate | 0.89 | 0.76 | 0.82 | 19,335 |

Macro recall is 0.71 and macro F1 is 0.64. The dataset is heavily imbalanced (Moderate ≈ 74% of the test set), so **class weighting** (`balanced_subsample` / sample weights) was used to prioritize catching Major and Minor interactions instead of maximizing raw accuracy.

**Most influential features:** drug frequency, DDInter ID frequency, and graph metrics (degree centrality, pair centrality product, pair total degree). The knowledge-graph features rank among the top predictors, which supports the graph-based approach.

## 🕸️ Knowledge Graph

- **1,902 nodes** (drugs and conditions) and **130,422 edges** (interactions and contraindications), built with NetworkX.
- Extracted structural features per drug pair: degree, degree centrality, pair total degree, and pair centrality product.

## 🗂️ Data

- **DDInter** interaction records: 222,383 raw rows → removed **62,148 duplicates** and **29,813 rows with unknown severity** → **130,422 clean pairs**.
- **Drug information database:** 718 drugs × 13 attributes (class, indications, contraindications, side effects, pregnancy category, and more).

## ⚙️ Pipeline

1. Load and merge the interaction and drug-info datasets.
2. Clean: duplicates, text normalization (strip / lowercase), remove `unknown` severity labels.
3. EDA: severity distribution, most-interacting drugs, dosage forms, indications.
4. Build the knowledge graph and extract graph metrics.
5. Feature engineering: **18 structured features** (same-class flag, pregnancy-risk ordinal mapping, text lengths, frequency encodings, graph metrics) + **100 TF-IDF features** from indications and contraindications = 118 features.
6. Stratified split, standard scaling, class-imbalance handling.
7. Train and compare Random Forest, XGBoost, CatBoost, and LightGBM; analyze feature importance.
8. Save the champion model, scaler, label encoder, and TF-IDF vectorizer with Joblib for the app.

## ✨ App Features

- **Real-time DDI severity prediction** for a selected drug pair.
- **Drug safety lookup:** class, indications, contraindications, side effects, pregnancy category.
- **Clinical entity linking (NLP):** maps medical terms in free-text notes to ICD-10 and RxNorm codes.
- **PDF prescription generator** with a QR code for pharmacy verification.

## 🛠️ Tech Stack

Streamlit · Scikit-Learn · XGBoost · CatBoost · LightGBM · NetworkX · Pandas · NumPy · Matplotlib · Seaborn · Joblib · ReportLab · QR code library

## 📁 Repository Structure

```
Drug_Knowledge_Base/
├── app.py                          # Streamlit web application
├── drug_knowledge_pipline.ipynb    # EDA, graph building, feature engineering, training
├── datasets/                       # DDInter interaction files + drug info database
├── saved_models/                   # Champion model + scaler + encoder + TF-IDF (Joblib)
├── requirements.txt
└── .gitignore
```

## 🚀 Quickstart

```bash
git clone https://github.com/habiba-10/Drug_Knowledge_Base.git
cd Drug_Knowledge_Base
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- **Possible optimistic bias:** frequency encodings and knowledge-graph metrics were computed on the full dataset before the train/test split. Recomputing them on the training split only is the first planned fix, and the reported test scores should be read with that in mind.
- Minor-class precision is low (0.40); many Moderate pairs are predicted as Minor or Major.
- Predictions are based on a research dataset and **must not replace clinical judgment or a pharmacist's review**. This is an educational prototype, not a medical device.

## 🔭 Future Work

- Compute graph and frequency features on the training split only, then re-evaluate.
- Try SMOTE / focal loss and threshold tuning for the Minor and Major classes.
- Add stratified cross-validation and hyperparameter search.
- Explore graph embeddings (Node2Vec) in place of hand-crafted degree features.

## 👤 Author

**Habiba Ezat** · [GitHub](https://github.com/habiba-10) · [LinkedIn](https://www.linkedin.com/in/habiba-ezat-a39521293)
