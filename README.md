# 🩺 Drug Knowledge Base & Expert System

An AI-powered clinical expert system designed for real-time analysis of Drug-Drug Interactions (DDI) and chronic condition contraindications. The platform enables healthcare providers to evaluate medication safety, perform NLP-driven medical entity linking, and generate verified digital prescriptions.

---

## 📸 Overview & Key Features

- **Real-Time Drug-Drug Interaction (DDI) Engine**: Predicts interaction severity levels (`Major`, `Moderate`, `Minor`, `Safe`) using a trained Machine Learning pipeline.
- **Knowledge Graph Integration**: Models complex relationships between drugs, therapeutic indications, and contraindications using NetworkX.
- **Clinical Entity Linking (NLP)**: Automatically detects medical entities, symptoms, and conditions from unstructured notes and maps them to standard clinical coding standards (**ICD-10** and **RxNorm**).
- **Interactive Web Dashboard**: Built with Streamlit, featuring real-time visual risk badges and clinical safety guidelines.
- **Official PDF Prescription Generator**: Automatically exports formatted medical prescriptions complete with an official QR code for pharmacy verification.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend / Application Framework**: Streamlit
- **Machine Learning & Pipeline**: Scikit-Learn, XGBoost, CatBoost, LightGBM, Joblib
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Graph & Network Processing**: NetworkX
- **Visualization**: Matplotlib, Seaborn
- **Document Generation**: ReportLab, PyQRCode / QRCoder

---

## 📁 Repository Structure

```text
Drug_Knowledge_Base/
├── app.py                            # Main Streamlit web application
├── drug_knowledge_pipeline.ipynb     # Pipeline notebook (EDA, Graph Building, Model Training)
├── datasets/                         # Medical datasets & interaction records
│   ├── ddinter_downloads_code_*.csv
│   └── Drug finder db w_o brands - deepseek_csv_20250915_dff2b8.csv
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Standard Git ignore rules
