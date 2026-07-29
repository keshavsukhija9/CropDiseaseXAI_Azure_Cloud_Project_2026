import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
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

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
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

  return (
    <div className="app">
      <header className="header">
        <h1>🌿 Crop Disease XAI</h1>
        <p>Explainable AI Cloud Platform for Precision Crop Disease Diagnosis</p>
      </header>

      <main className="main-grid">
        <section className="upload-card">
          <h2>Diagnose a Leaf</h2>
          <label className="file-drop">
            <input type="file" accept="image/*" onChange={handleFileChange} hidden />
            {preview ? <img src={preview} alt="preview" className="preview-img" /> : <span>Click to choose a leaf image</span>}
          </label>
          <button className="diagnose-btn" onClick={handleSubmit} disabled={!file || loading}>
            {loading ? "Analyzing..." : "Diagnose"}
          </button>
          {error && <p className="error-text">Error: {error}</p>}
          {result && (
            <div className="result-card">
              <div className="result-row">
                <span className="label">Prediction</span>
                <span className="value">{result.predicted_class.replaceAll("___", " — ").replaceAll("_", " ")}</span>
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
                <span className="value">{result.severity_pct}%</span>
              </div>
              <div className="treatment-box">
                <strong>Recommended action:</strong>
                <p>{result.treatment}</p>
              </div>
              <img src={`${API_BASE}${result.explanation_url}`} alt="explanation heatmap" className="heatmap-img" />
              <p className="caption">Heatmap shows the leaf regions that drove this diagnosis.</p>
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
                  <span className="history-class">{h.predicted_class.replaceAll("___", " — ").replaceAll("_", " ")}</span>
                  <span className="history-conf">{(h.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="history-meta">
                  <span>severity {h.severity_pct}%</span>
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
