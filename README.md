# IntentIQ — E-commerce purchase intent

IntentIQ is an end-to-end purchase-intent demo built exclusively from
`data/dataset_04_ecommerce_purchase_intent.csv`. It includes reproducible
analysis/training, a Flask prediction API, and a React dashboard for session
predictions and model analytics.

## Results

The training script uses an 80/20 **stratified** split (`random_state=42`).
`StandardScaler` and Logistic Regression are fitted together in one sklearn
Pipeline, after the split, preventing test-set leakage.

| Metric | Held-out result |
| --- | ---: |
| Accuracy | 0.7300 (73.0%) |
| Precision | 0.7396 (74.0%) |
| Recall | 0.7100 (71.0%) |
| F1 | 0.7245 (72.4%) |
| ROC AUC | 0.8004 |

The dataset contains 1,000 sessions, six input features, and balanced target
classes (500 no-purchase / 500 purchase). The original CSV is preserved
unchanged (SHA-256:
`CF52C4B18F5DCEAEA19C423D2D6F92C5912424B0F3257FE81C0D27527D1F7E2D`).

## Project layout

```text
data/       Source CSV (do not modify)
notebooks/  analysis.py — EDA, training, evaluation
model/      model.joblib, metrics.json, feature metadata, PNG plots
backend/    Flask API (/health, /predict, /analytics)
frontend/   Vite + React dashboard
```

## Run locally

From the project root, install the Python dependencies and generate the
artifacts:

```powershell
python -m pip install -r requirements.txt
python notebooks\analysis.py
```

Start the API (in a separate terminal):

```powershell
python backend\app.py
```

Start the React app (in another terminal):

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally
`http://localhost:5173`). The frontend expects the API at
`http://127.0.0.1:5000`; override it with `VITE_API_URL` if needed.

### API example

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:5000/predict `
  -ContentType "application/json" `
  -Body '{"pages_viewed":54,"session_minutes":13.29,"products_viewed":2,"cart_additions":5,"discount_seen":2,"previous_orders":5}'
```

`GET /analytics` returns the generated metrics, confusion matrix, feature
coefficients, dataset summary, and training configuration.
