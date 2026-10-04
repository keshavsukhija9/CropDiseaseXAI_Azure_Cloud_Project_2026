import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [leafFile, setLeafFile] = useState(null);
  const [uavFile, setUavFile] = useState(null);
  const [leafPreview, setLeafPreview] = useState(null);
  const [uavPreview, setUavPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      setHistory(data);
    } catch (e) {}
  };

  useEffect(() => { fetchHistory(); }, []);

  const handleLeafChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setLeafFile(selected);
    setLeafPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handleUavChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setUavFile(selected);
    setUavPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!leafFile || !uavFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("leaf_image", leafFile);
    formData.append("uav_image", uavFile);
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      fetchHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const severityColor = (pct) => {
    if (pct < 15) return "#4ade80";
    if (pct < 40) return "#facc15";
    return "#f87171";
  };

  const decisionColor = (decision) => {
    if (decision === "Automatically accepted") return "#4ade80";
    if (decision === "Agronomist review required") return "#f87171";
    return "#facc15";
  };

  const prettify = (s) => s.replaceAll("___", " — ").replaceAll("_", " ");

  return (
    <div className="app">
      <header className="header">
        <h1>🌿 Crop Disease XAI</h1>
        <p>Cross-Scale Explainable AI Cloud Framework for UAV-Based Crop Disease Detection, Severity Estimation and Field-Level Decision Support</p>
      </header>

      <main className="main-grid">
        <section className="upload-card">
          <h2>Diagnose (Leaf + UAV)</h2>
          <div style={{ display: "flex", gap: "12px" }}>
            <label className="file-drop" style={{ flex: 1 }}>
              <input type="file" accept="image/*" onChange={handleLeafChange} hidden />
              {leafPreview ? <img src={leafPreview} alt="leaf preview" className="preview-img" /> : <span>Leaf image (micro)</span>}
            </label>
            <label className="file-drop" style={{ flex: 1 }}>
              <input type="file" accept="image/*" onChange={handleUavChange} hidden />
              {uavPreview ? <img src={uavPreview} alt="uav preview" className="preview-img" /> : <span>UAV field image (macro)</span>}
            </label>
          </div>
          <button className="diagnose-btn" onClick={handleSubmit} disabled={!leafFile || !uavFile || loading}>
            {loading ? "Analyzing..." : "Diagnose"}
          </button>
          {error && <p className="error-text">Error: {error}</p>}
          {result && (
            <div className="result-card">
              <div className="result-row">
                <span className="label">Prediction</span>
                <span className="value">{prettify(result.predicted_class)}</span>
              </div>
              <div className="result-row">
                <span className="label">Confidence</span>
                <span className="value">{(result.confidence * 100).toFixed(2)}%</span>
              </div>
              <div className="result-row">
                <span className="label">Severity</span>
                <div className="severity-bar-track">
                  <div className="severity-bar-fill" style={{ width: `${result.severity_pct}%`, background: severityColor(result.severity_pct) }} />
                </div>
                <span className="value">{result.severity_pct}% ({result.severity_bucket})</span>
              </div>
              <div className="result-row">
                <span className="label">Uncertainty</span>
                <span className="value">{result.uncertainty}</span>
              </div>
              <div className="result-row">
                <span className="label">Decision</span>
                <span className="value" style={{ color: decisionColor(result.decision), fontWeight: 600 }}>
                  {result.decision}
                </span>
              </div>
              <div className="treatment-box">
                <strong>Recommended action:</strong>
                <p>{result.treatment}</p>
              </div>
              <img src={`${API_BASE}${result.explanation_url}`} alt="explanation heatmap" className="heatmap-img" />
              <p className="caption">Grad-CAM heatmap shows the leaf regions that drove this diagnosis.</p>
            </div>
          )}
        </section>

        <section className="history-card">
          <h2>Recent Diagnoses</h2>
          {history.length === 0 && <p className="empty-text">No diagnoses yet.</p>}
          <ul className="history-list">
            {history.map((h) => (
              <li key={h.id} className="history-item">
                <div className="history-main">
                  <span className="history-class">{prettify(h.predicted_class)}</span>
                  <span className="history-conf">{(h.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="history-meta">
                  <span>severity {h.severity_pct}% ({h.severity_bucket})</span>
                  <span style={{ color: decisionColor(h.decision) }}>{h.decision}</span>
                  <span>{new Date(h.timestamp).toLocaleString()}</span>
                </div>
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}

export default App;
