import { useCallback, useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

const INITIAL_FORM = {
  gender: "Female",
  SeniorCitizen: "0",
  Partner: "Yes",
  Dependents: "No",
  tenure: "18",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: "84.7",
  TotalCharges: "1524.6",
};

const FIELD_GROUPS = [
  {
    id: "profile",
    title: "Customer profile",
    description: "Basic details associated with the account.",
    icon: "user",
    fields: [
      { name: "gender", label: "Gender", type: "select", options: ["Female", "Male"] },
      { name: "SeniorCitizen", label: "Senior citizen", type: "select", options: [{ value: "0", label: "No" }, { value: "1", label: "Yes" }] },
      { name: "Partner", label: "Has a partner", type: "select", options: ["Yes", "No"] },
      { name: "Dependents", label: "Has dependents", type: "select", options: ["Yes", "No"] },
      { name: "tenure", label: "Tenure", type: "number", min: 0, max: 72, suffix: "months", help: "Months the customer has stayed." },
    ],
  },
  {
    id: "services",
    title: "Services & support",
    description: "Current services active on the account.",
    icon: "layers",
    fields: [
      { name: "PhoneService", label: "Phone service", type: "select", options: ["Yes", "No"] },
      { name: "MultipleLines", label: "Multiple lines", type: "select", options: ["Yes", "No", "No phone service"] },
      { name: "InternetService", label: "Internet service", type: "select", options: ["DSL", "Fiber optic", "No"] },
      { name: "OnlineSecurity", label: "Online security", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "OnlineBackup", label: "Online backup", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "DeviceProtection", label: "Device protection", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "TechSupport", label: "Tech support", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "StreamingTV", label: "Streaming TV", type: "select", options: ["Yes", "No", "No internet service"] },
      { name: "StreamingMovies", label: "Streaming movies", type: "select", options: ["Yes", "No", "No internet service"] },
    ],
  },
  {
    id: "billing",
    title: "Contract & billing",
    description: "Commercial terms and monthly charges.",
    icon: "wallet",
    fields: [
      { name: "Contract", label: "Contract type", type: "select", options: ["Month-to-month", "One year", "Two year"] },
      { name: "PaperlessBilling", label: "Paperless billing", type: "select", options: ["Yes", "No"] },
      { name: "PaymentMethod", label: "Payment method", type: "select", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] },
      { name: "MonthlyCharges", label: "Monthly charges", type: "number", min: 0.01, step: "0.01", prefix: "$" },
      { name: "TotalCharges", label: "Total charges", type: "number", min: 0, step: "0.01", prefix: "$" },
    ],
  },
];

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="2" /><rect x="14" y="3" width="7" height="7" rx="2" /><rect x="3" y="14" width="7" height="7" rx="2" /><rect x="14" y="14" width="7" height="7" rx="2" /></>,
    chart: <><path d="M3 3v18h18" /><path d="m7 14 4-4 3 3 6-7" /></>,
    user: <><circle cx="12" cy="8" r="4" /><path d="M4.5 21a7.5 7.5 0 0 1 15 0" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5" /><path d="m3 16 9 5 9-5" /></>,
    wallet: <><rect x="3" y="5" width="18" height="15" rx="3" /><path d="M3 9h18" /><path d="M16 14h2" /><path d="M7 5V3h10v2" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    refresh: <><path d="M20 7v5h-5" /><path d="M4 17v-5h5" /><path d="M5.6 9A7 7 0 0 1 18 6l2 1" /><path d="M4 17l2 1a7 7 0 0 0 12.4-3" /></>,
    pulse: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    server: <><rect x="3" y="3" width="18" height="7" rx="2" /><rect x="3" y="14" width="18" height="7" rx="2" /><path d="M7 6.5h.01M7 17.5h.01" /></>,
    chevron: <path d="m7 10 5 5 5-5" />,
    alert: <><path d="M10.3 3.9 2.5 17.4A2 2 0 0 0 4.2 20h15.6a2 2 0 0 0 1.7-2.6L13.7 3.9a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4M12 17h.01" /></>,
  };
  return <svg {...common}>{paths[name] || paths.spark}</svg>;
}

function Field({ field, value, onChange }) {
  const id = `field-${field.name}`;
  const options = field.options || [];
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{field.label}</span>
      <span className={`input-wrap ${field.prefix ? "has-prefix" : ""} ${field.suffix ? "has-suffix" : ""}`}>
        {field.prefix && <span className="input-affix prefix">{field.prefix}</span>}
        {field.type === "select" ? (
          <select id={id} name={field.name} value={value} onChange={onChange}>
            {options.map((option) => {
              const item = typeof option === "string" ? { value: option, label: option } : option;
              return <option key={item.value} value={item.value}>{item.label}</option>;
            })}
          </select>
        ) : (
          <input
            id={id}
            name={field.name}
            type="number"
            min={field.min}
            max={field.max}
            step={field.step || "1"}
            value={value}
            onChange={onChange}
            required
          />
        )}
        {field.type === "select" && <span className="select-chevron"><Icon name="chevron" size={15} /></span>}
        {field.suffix && <span className="input-affix suffix">{field.suffix}</span>}
      </span>
      {field.help && <span className="field-help">{field.help}</span>}
    </label>
  );
}

function RiskRing({ probability }) {
  const percent = Math.round(probability * 100);
  return (
    <div className="risk-ring" style={{ "--risk-angle": `${percent * 3.6}deg` }}>
      <div className="risk-ring-inner">
        <span className="risk-percent">{percent}<small>%</small></span>
        <span className="risk-caption">churn score</span>
      </div>
    </div>
  );
}

function ResultPanel({ result, onReset }) {
  const isChurn = result.predicted_state === "Churn";
  const percent = Math.round(result.churn_probability * 100);
  const severity = percent >= 70 ? "High" : percent >= 40 ? "Moderate" : "Lower";
  return (
    <section className={`result-card ${isChurn ? "result-risk" : "result-safe"}`} aria-live="polite">
      <div className="result-heading">
        <div className="result-icon"><Icon name={isChurn ? "pulse" : "shield"} size={22} /></div>
        <div>
          <p className="eyebrow">PREDICTION COMPLETE</p>
          <h2>{isChurn ? "Churn signal detected" : "Customer likely to stay"}</h2>
        </div>
        <span className={`status-pill ${isChurn ? "status-warn" : "status-good"}`}>
          {isChurn ? "At risk" : "Lower risk"}
        </span>
      </div>
      <div className="result-body">
        <div className="ring-column">
          <RiskRing probability={result.churn_probability} />
          <span className="ring-undertext">Model probability</span>
        </div>
        <div className="result-details">
          <div className="result-summary">
            <span className="summary-label">Predicted state</span>
            <strong>{result.predicted_state}</strong>
            <span className="summary-sub">Decision threshold: {(result.decision_threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="detail-grid">
            <div className="detail-metric">
              <span>Risk band</span>
              <strong>{severity}</strong>
            </div>
            <div className="detail-metric">
              <span>Model</span>
              <strong>{result.model_name.replace("Classifier", "")}</strong>
            </div>
          </div>
          <div className="result-note">
            <Icon name="alert" size={15} />
            <span>This model classifies the supplied <strong>StateNextMonth</strong> label. Its source and temporal alignment have not been independently verified.</span>
          </div>
        </div>
      </div>
      <div className="result-actions">
        <button className="button button-secondary" onClick={onReset} type="button"><Icon name="refresh" size={16} /> New prediction</button>
        <span className="result-footnote"><Icon name="check" size={14} /> Preprocessing + model API connected</span>
      </div>
    </section>
  );
}

export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  const checkApi = useCallback(async () => {
    setApiStatus("checking");
    try {
      const response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(3500) });
      if (!response.ok) throw new Error("Backend health check failed.");
      const data = await response.json();
      setApiStatus(data.model_loaded ? "online" : "no-model");
    } catch {
      setApiStatus("offline");
    }
  }, []);

  useEffect(() => { checkApi(); }, [checkApi]);

  const updateField = (event) => {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    setError("");
  };

  const serviceCount = useMemo(() =>
    ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
      .reduce((count, name) => count + (form[name] === "Yes" ? 1 : 0), 0),
  [form]);

  const predict = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    const payload = {
      ...form,
      SeniorCitizen: Number(form.SeniorCitizen),
      tenure: Number(form.tenure),
      MonthlyCharges: Number(form.MonthlyCharges),
      TotalCharges: Number(form.TotalCharges),
    };
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = typeof data.detail === "string" ? data.detail : "Prediction failed. Check the API and input values.";
        throw new Error(message);
      }
      setResult(data);
      setApiStatus("online");
      window.setTimeout(() => document.getElementById("prediction-result")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (err) {
      if (err.name === "TimeoutError" || err.name === "AbortError") {
        setError("The API took too long to respond. Check that the backend is running and try again.");
      } else {
        setError(err.message || "Could not connect to the prediction API.");
      }
      setApiStatus("offline");
    } finally {
      setLoading(false);
    }
  };

  const resetPrediction = () => {
    setForm(INITIAL_FORM);
    setResult(null);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const apiLabel = {
    checking: "Checking API",
    online: "API connected",
    "no-model": "API online · model missing",
    offline: "API unavailable",
  }[apiStatus];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="ChurnSense home">
          <span className="brand-mark"><Icon name="pulse" size={22} /></span>
          <span className="brand-name">churn<span>sense</span><small>Customer intelligence</small></span>
        </a>
        <div className="side-section-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-link active" href="#overview"><Icon name="grid" /> <span>Overview</span><span className="nav-active-dot" /></a>
          <a className="nav-link" href="#customer-form"><Icon name="user" /> <span>Customer analysis</span></a>
          <a className="nav-link" href="#how-it-works"><Icon name="chart" /> <span>How it works</span></a>
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-callout">
          <div className="callout-orb"><Icon name="spark" size={18} /></div>
          <strong>Make every signal count.</strong>
          <p>Use model predictions as one input to customer-retention decisions.</p>
          <a href="#how-it-works">Explore the workflow <Icon name="arrow" size={14} /></a>
        </div>
        <div className="sidebar-footer"><span className="footer-dot" /> Machine learning workspace <span className="footer-version">v1.0</span></div>
      </aside>

      <main className="main-content" id="top">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-slash">/</span><strong>Customer analysis</strong></div>
          <div className={`api-status api-${apiStatus}`}><span className="api-status-dot" />{apiLabel}<button className="icon-button" title="Check API connection again" onClick={checkApi} type="button"><Icon name="refresh" size={15} /></button></div>
        </header>

        <section className="hero" id="overview">
          <div className="hero-copy">
            <div className="hero-kicker"><span className="kicker-line" /> CUSTOMER RETENTION INTELLIGENCE</div>
            <h1>Understand your<br /><span>customer signals.</span></h1>
            <p>Bring customer, service and billing details together to explore the model's churn classification and support more informed retention decisions.</p>
            <a className="hero-link" href="#customer-form">Analyze a customer <Icon name="arrow" size={16} /></a>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="art-halo halo-one" />
            <div className="art-halo halo-two" />
            <div className="art-orbit orbit-one" />
            <div className="art-orbit orbit-two" />
            <div className="art-core"><Icon name="pulse" size={42} /></div>
            <div className="art-chip chip-top"><span className="chip-dot lilac" /> Predictive signals</div>
            <div className="art-chip chip-bottom"><span className="chip-dot mint" /> Connected pipeline</div>
            <div className="art-node node-one" />
            <div className="art-node node-two" />
            <div className="art-node node-three" />
          </div>
        </section>

        <section className="quick-stats" aria-label="Dataset facts">
          <article className="stat-card">
            <span className="stat-icon lavender"><Icon name="user" size={18} /></span>
            <div><span className="stat-label">Dataset size</span><strong>7,043 <small>customers</small></strong><span className="stat-note">Selected project dataset</span></div>
          </article>
          <article className="stat-card">
            <span className="stat-icon aqua"><Icon name="layers" size={18} /></span>
            <div><span className="stat-label">Input features</span><strong>19 <small>raw fields</small></strong><span className="stat-note">Profile, service & billing</span></div>
          </article>
          <article className="stat-card">
            <span className="stat-icon peach"><Icon name="shield" size={18} /></span>
            <div><span className="stat-label">Prediction target</span><strong>StateNextMonth <small>label</small></strong><span className="stat-note">Label provenance unverified</span></div>
          </article>
        </section>

        <section className="analysis-heading" id="customer-form">
          <div>
            <p className="eyebrow">CUSTOMER WORKSPACE</p>
            <h2>Run a customer analysis</h2>
            <p className="section-subtitle">Enter the customer details below. The connected API applies the shared feature pipeline and returns the model probability.</p>
          </div>
          <button className="text-button" type="button" onClick={resetPrediction}><Icon name="refresh" size={15} /> Reset form</button>
        </section>

        <div className="workspace-grid">
          <form className="form-column" onSubmit={predict}>
            {FIELD_GROUPS.map((group, groupIndex) => (
              <section className="form-card" key={group.id}>
                <div className="form-card-heading">
                  <div className="form-card-icon"><Icon name={group.icon} size={19} /></div>
                  <div><h3>{group.title}</h3><p>{group.description}</p></div>
                  <span className="group-number">0{groupIndex + 1}</span>
                </div>
                <div className={`field-grid ${group.id === "services" ? "service-grid" : ""}`}>
                  {group.fields.map((field) => <Field key={field.name} field={field} value={form[field.name]} onChange={updateField} />)}
                </div>
              </section>
            ))}
            {error && <div className="error-banner" role="alert"><Icon name="alert" size={18} /><div><strong>Prediction couldn't be completed</strong><span>{error}</span></div></div>}
            <button className="button button-primary predict-button" type="submit" disabled={loading}>
              {loading ? <><span className="spinner" /> Analyzing customer…</> : <>Generate prediction <Icon name="arrow" size={17} /></>}
            </button>
            <p className="submit-footnote"><Icon name="shield" size={14} /> Inputs are sent to your configured API. No customer record is stored by this frontend.</p>
          </form>

          <aside className="insight-column">
            <div className="insight-card">
              <div className="insight-header"><div className="insight-icon"><Icon name="pulse" size={19} /></div><span className="insight-live">LIVE WORKFLOW</span></div>
              <h3>From customer details<br />to model output.</h3>
              <p>Every prediction follows the same preprocessing and inference pipeline as the project model.</p>
              <div className="workflow-list">
                <div className="workflow-step"><span className="workflow-number">01</span><div><strong>Customer inputs</strong><small>Validate raw service & billing fields</small></div><Icon name="check" size={16} /></div>
                <div className="workflow-connector" />
                <div className="workflow-step"><span className="workflow-number">02</span><div><strong>Feature pipeline</strong><small>Engineer, encode & scale values</small></div><Icon name="check" size={16} /></div>
                <div className="workflow-connector" />
                <div className="workflow-step"><span className="workflow-number">03</span><div><strong>Model inference</strong><small>Calculate positive-class probability</small></div><Icon name="check" size={16} /></div>
              </div>
              <div className="workflow-foot"><span className="footer-dot" /> Shared backend pipeline</div>
            </div>
            <div className="service-snapshot">
              <div className="snapshot-heading"><span>Service snapshot</span><span className="snapshot-total">{serviceCount} active signals</span></div>
              <div className="snapshot-meter"><span style={{ width: `${(serviceCount / 8) * 100}%` }} /></div>
              <p>Based on the current form, {serviceCount} of 8 service indicators are marked “Yes”. This is descriptive only, not a separate risk score.</p>
            </div>
            <div className="methodology-note"><Icon name="alert" size={16} /><p><strong>Interpret results carefully.</strong> The model predicts the supplied StateNextMonth label. Its source and temporal alignment have not been independently verified.</p></div>
          </aside>
        </div>

        {result && <div id="prediction-result" className="result-anchor"><ResultPanel result={result} onReset={resetPrediction} /></div>}

        <section className="how-section" id="how-it-works">
          <div><p className="eyebrow">METHODOLOGY</p><h2>Designed for responsible analysis</h2><p className="section-subtitle">A prediction is one piece of evidence—not a reason to act on a customer without context.</p></div>
          <div className="principle-grid">
            <article><span className="principle-icon"><Icon name="shield" /></span><h3>Consistent processing</h3><p>The same feature engineering and fitted preprocessing pipeline are reused for model inference.</p></article>
            <article><span className="principle-icon"><Icon name="chart" /></span><h3>Transparent probability</h3><p>See the predicted State, probability and threshold that produced the classification.</p></article>
            <article><span className="principle-icon"><Icon name="user" /></span><h3>Human-led decisions</h3><p>Use predictions alongside customer context, uncertainty and appropriate business review.</p></article>
          </div>
        </section>

        <footer className="page-footer"><a className="footer-brand" href="#top"><span className="brand-mark mini"><Icon name="pulse" size={16} /></span> churnsense</a><span>Customer churn classification workspace</span><span>Powered by a connected ML API</span></footer>
      </main>
    </div>
  );
}
