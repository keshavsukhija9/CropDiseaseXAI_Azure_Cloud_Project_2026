import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Crop Disease Diagnosis</h1>
      <p>Upload a leaf image to get a diagnosis with an explainability heatmap.</p>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      {preview && (
        <div style={{ marginTop: 16 }}>
          <img src={preview} alt="preview" style={{ maxWidth: 300, borderRadius: 8 }} />
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        <button onClick={handleSubmit} disabled={!file || loading}>
          {loading ? "Diagnosing..." : "Diagnose"}
        </button>
      </div>

      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: 24, padding: 16, border: "1px solid #ddd", borderRadius: 8 }}>
          <h3>Result</h3>
          <p><strong>Prediction:</strong> {result.predicted_class}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          <img
            src={`http://127.0.0.1:8000${result.explanation_url}`}
            alt="explanation heatmap"
            style={{ maxWidth: 300, borderRadius: 8, marginTop: 8 }}
          />
          <p style={{ fontSize: 12, color: "#666" }}>Heatmap shows the regions that drove this diagnosis.</p>
        </div>
      )}
    </div>
  );
}

export default App;
