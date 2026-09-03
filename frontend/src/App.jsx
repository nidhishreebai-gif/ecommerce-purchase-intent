import { useEffect, useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const fields = [
  { key: "pages_viewed", label: "Pages viewed", hint: "18–65", step: 1 },
  { key: "session_minutes", label: "Session duration", hint: "Minutes", step: 0.01 },
  { key: "products_viewed", label: "Products viewed", hint: "0–10", step: 1 },
  { key: "cart_additions", label: "Cart additions", hint: "0–10", step: 1 },
  { key: "discount_seen", label: "Discounts seen", hint: "0–9", step: 1 },
  { key: "previous_orders", label: "Previous orders", hint: "0–9", step: 1 },
];

const defaultForm = {
  pages_viewed: 40,
  session_minutes: 36.7,
  products_viewed: 3,
  cart_additions: 3,
  discount_seen: 3,
  previous_orders: 3,
};

function Metric({ label, value, suffix = "" }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}{suffix}</strong>
    </div>
  );
}

function App() {
  const [activeView, setActiveView] = useState("predict");
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/analytics`)
      .then((response) => {
        if (!response.ok) throw new Error("Analytics unavailable");
        return response.json();
      })
      .then(setAnalytics)
      .catch(() => setError("Connect the Flask API to load live model analytics."));
  }, []);

  const maxImportance = useMemo(
    () => Math.max(...(analytics?.feature_importance || []).map((item) => item.absolute_coefficient), 1),
    [analytics],
  );

  const updateField = (key, value) => {
    setForm((current) => ({ ...current, [key]: value === "" ? "" : Number(value) }));
  };

  const submitPrediction = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    const payload = Object.fromEntries(
      fields.map(({ key }) => [key, Number(form[key])]),
    );
    if (Object.values(payload).some((value) => !Number.isFinite(value))) {
      setError("Enter a valid number for every session signal.");
      setLoading(false);
      return;
    }
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.details?.join(" ") || "Prediction failed.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Prediction failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">IQ</div>
          <div><strong>IntentIQ</strong><small>Commerce intelligence</small></div>
        </div>
        <div className="sidebar-copy">
          <p className="eyebrow">MODEL CONSOLE</p>
          <h1>Know the next best action.</h1>
          <p>Turn live browsing signals into a confident purchase-intent read.</p>
        </div>
        <nav>
          <button className={activeView === "predict" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("predict")}>
            <span>⌁</span> Predict intent
          </button>
          <button className={activeView === "analytics" ? "nav-item active" : "nav-item"} onClick={() => setActiveView("analytics")}>
            <span>◌</span> Model analytics
          </button>
        </nav>
        <div className="sidebar-footer"><span className="status-dot" /> API ready for local use</div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div><p className="eyebrow">PURCHASE INTENT / OVERVIEW</p><h2>{activeView === "predict" ? "Predict customer intent" : "Model analytics"}</h2></div>
          <div className="model-pill"><span className="status-dot" /> Logistic Regression <b>•</b> v1.0</div>
        </header>

        {error && <div className="alert">{error}</div>}

        {activeView === "predict" ? (
          <section className="predict-layout">
            <div className="panel form-panel">
              <div className="panel-heading"><div><p className="eyebrow">SESSION SIGNALS</p><h3>Describe a browsing session</h3></div><span className="step-badge">01 / 01</span></div>
              <p className="muted">Adjust the signals below to estimate the likelihood of a purchase.</p>
              <form onSubmit={submitPrediction}>
                <div className="form-grid">
                  {fields.map((field) => (
                    <label className="field" key={field.key}>
                      <span>{field.label}<small>{field.hint}</small></span>
                      <input type="number" min={analytics?.dataset?.feature_summary?.[field.key]?.min} max={analytics?.dataset?.feature_summary?.[field.key]?.max} step={field.step} value={form[field.key]} onChange={(event) => updateField(field.key, event.target.value)} required />
                    </label>
                  ))}
                </div>
                <button className="primary-button" type="submit" disabled={loading}>{loading ? "Calculating…" : "Calculate intent"} <span>→</span></button>
              </form>
              <button className="text-button" onClick={() => { setForm(defaultForm); setResult(null); setError(""); }}>Reset to typical session</button>
            </div>

            <div className={result ? "panel result-panel has-result" : "panel result-panel"}>
              {!result ? (
                <div className="empty-result"><div className="empty-icon">✦</div><p className="eyebrow">YOUR RESULT</p><h3>Ready when you are</h3><p className="muted">Enter session details and calculate a prediction to see the recommended signal.</p></div>
              ) : (
                <div className="result-content">
                  <div className="result-top"><div><p className="eyebrow">PREDICTION RESULT</p><h3>{result.prediction}</h3></div><span className={result.prediction_code ? "outcome likely" : "outcome unlikely"}>{result.prediction_code ? "HIGH INTENT" : "LOW INTENT"}</span></div>
                  <div className="probability"><div className="probability-value">{Math.round(result.purchase_probability * 100)}<small>%</small></div><div><strong>Purchase probability</strong><p className="muted">No-purchase probability {Math.round(result.no_purchase_probability * 100)}%</p></div></div>
                  <div className="progress-track"><div className={result.prediction_code ? "progress-fill positive" : "progress-fill"} style={{ width: `${result.purchase_probability * 100}%` }} /></div>
                  <div className="recommendation"><span>↗</span><div><strong>{result.prediction_code ? "Prioritize this session" : "Nurture this session"}</strong><p>{result.prediction_code ? "Consider a personalized offer or a low-friction checkout prompt." : "Keep the experience helpful with relevant recommendations and a gentle reminder."}</p></div></div>
                </div>
              )}
            </div>
          </section>
        ) : (
          <section className="analytics-view">
            <div className="metrics-grid">
              <Metric label="Accuracy" value={analytics ? `${(analytics.metrics.accuracy * 100).toFixed(1)}%` : "—"} />
              <Metric label="ROC AUC" value={analytics ? analytics.metrics.roc_auc.toFixed(3) : "—"} />
              <Metric label="Test sessions" value={analytics?.training?.test_rows ?? "—"} />
              <Metric label="Training sessions" value={analytics?.training?.train_rows ?? "—"} />
            </div>
            <div className="analytics-grid">
              <div className="panel">
                <div className="panel-heading"><div><p className="eyebrow">DRIVERS</p><h3>What influences intent?</h3></div></div>
                <p className="muted">Standardized model coefficients show relative signal strength.</p>
                <div className="importance-list">{(analytics?.feature_importance || []).map((item) => <div className="importance-row" key={item.feature}><div className="importance-label"><span>{item.feature.replaceAll("_", " ")}</span><b className={item.coefficient >= 0 ? "positive-text" : "negative-text"}>{item.coefficient >= 0 ? "+" : ""}{item.coefficient.toFixed(2)}</b></div><div className="importance-track"><div className={item.coefficient >= 0 ? "importance-fill positive" : "importance-fill negative"} style={{ width: `${(item.absolute_coefficient / maxImportance) * 100}%` }} /></div></div>)}</div>
              </div>
              <div className="panel visual-panel">
                <div className="panel-heading"><div><p className="eyebrow">ERRORS</p><h3>Confusion matrix</h3></div></div>
                {analytics && <div className="confusion-matrix">
                  <span></span><b>Predicted no purchase</b><b>Predicted purchase</b>
                  <b>Actual no purchase</b><strong>{analytics.confusion_matrix[0][0]}</strong><strong>{analytics.confusion_matrix[0][1]}</strong>
                  <b>Actual purchase</b><strong>{analytics.confusion_matrix[1][0]}</strong><strong>{analytics.confusion_matrix[1][1]}</strong>
                </div>}
              </div>
              <div className="panel visual-panel">
                <div className="panel-heading"><div><p className="eyebrow">DISCRIMINATION</p><h3>ROC curve</h3></div></div>
                {analytics && <svg className="roc-chart" viewBox="0 0 300 210" role="img" aria-label="ROC curve">
                  <line x1="35" y1="175" x2="275" y2="175" />
                  <line x1="35" y1="175" x2="35" y2="20" />
                  <line className="baseline" x1="35" y1="175" x2="275" y2="20" />
                  <polyline points={analytics.roc_curve.false_positive_rate.map((x, index) => `${35 + x * 240},${175 - analytics.roc_curve.true_positive_rate[index] * 155}`).join(" ")} />
                </svg>}
              </div>
              <div className="panel">
                <div className="panel-heading"><div><p className="eyebrow">QUALITY CHECK</p><h3>Held-out performance</h3></div></div>
                <div className="quality-list">
                  {analytics && [["Precision", analytics.metrics.precision], ["Recall", analytics.metrics.recall], ["F1 score", analytics.metrics.f1]].map(([label, value]) => <div className="quality-row" key={label}><span>{label}</span><strong>{(value * 100).toFixed(1)}%</strong><div className="mini-track"><i style={{ width: `${value * 100}%` }} /></div></div>)}
                </div>
                <div className="dataset-note"><strong>{analytics?.dataset?.rows ?? "—"} sessions analyzed</strong><span>Balanced classes · stratified split · seed 42</span></div>
              </div>
            </div>
          </section>
        )}
        <footer>IntentIQ · Built for thoughtful, data-informed commerce experiences</footer>
      </main>
    </div>
  );
}

export default App;
