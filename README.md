# DiaPredict (Group02)

A 3-tier desktop application that assesses diabetes risk from standard
health metrics using a trained machine learning model. Built for
CPE178P-FOPI01 by David Charles B. Clemente and Nicole Jhune B. Porras.

## Structure

├── app.py                      # Tkinter desktop app (Presentation + Logic tiers)

├── model_training.ipynb        # Preprocessing, SMOTE augmentation, model

│                                #   training/comparison, and DB schema init

├── diabetes.csv                # Pima Indians Diabetes Database (768 records)

├── diapredict_best_model.pkl   # Serialized final model (Random Forest)

├── scaler.pkl                  # Serialized StandardScaler used at inference

├── diapredict.db               # SQLite database (users + assessments tables)

├── screenshots/                # App screenshots (login, form, result, history)

└── requirements.txt

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

The model, scaler, and database are already included and trained/initialized
— you can run the app directly:

```bash
python app.py
```

If you want to retrain the model from scratch (e.g. after changing
preprocessing), open and run `model_training.ipynb` top to bottom. It will
regenerate `scaler.pkl`, `diapredict_best_model.pkl`, and (re)create
`diapredict.db` with empty `users`/`assessments` tables.

## Model Summary

Two models were trained and compared on the Pima Indians Diabetes Database,
with SMOTE applied to the training set to address class imbalance
(400 negative / 214 positive → 400/400):

| Model | CV Accuracy (k=5) | Test Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.739 | 0.714 | 0.621 | 0.812 |
| **Random Forest** | **0.823** | **0.753** | **0.672** | **0.814** |

Random Forest was selected as the final model (highest F1-score and
AUC-ROC) and is what `app.py` loads at runtime.

## Known Limitations
- User passwords are stored in plaintext in the `users` table. Acceptable
  for a course project demo; a production version should hash credentials
  (e.g. with bcrypt) before storing them.
- Final test accuracy (75.3%) is below the original 80% target stated in
  the project proposal, though AUC-ROC (0.814) clears the 0.80 target. See
  the project documentation for a discussion of the precision/recall
  trade-off introduced by SMOTE rebalancing.
- `diapredict.db` included in this repo contains sample/test data created
  during development, not real patient records.

## App Screens
1. **Login / Registration** — email/password sign-in and account creation.
2. **Health Data Entry** — 8-field form (Glucose, BMI, Blood Pressure, Age,
   Insulin, Skin Thickness, Pregnancies, DPF score).
3. **Prediction Result** — risk probability, Low/Medium/High label, and a
   plain-language recommendation to consult a healthcare professional.
4. **Assessment History** — color-coded table of past assessments for the
   logged-in user, with CSV export.