import { useEffect, useState } from 'react'

const API = 'http://127.0.0.1:8000/api'

export default function App() {
  const [complaint, setComplaint] = useState('')
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState('gru')
  const [result, setResult] = useState(null)
  const [allResults, setAllResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiOnline, setApiOnline] = useState(false)

  useEffect(() => {
    fetch(`${API}/models`)
      .then(r => { if (!r.ok) throw new Error('API unavailable'); return r.json() })
      .then(data => { setModels(data.models || []); setApiOnline(true) })
      .catch(() => setApiOnline(false))
  }, [])

  async function submit(path) {
    if (complaint.trim().length < 3) {
      setError('Please enter a complaint with at least 3 characters.')
      return
    }
    setLoading(true); setError(''); setResult(null); setAllResults(null)
    try {
      const body = path.endsWith('/all')
        ? { complaint }
        : { complaint, model: selected }
      const response = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed')
      if (path.endsWith('/all')) setAllResults(data)
      else setResult(data)
    } catch (e) {
      setError(`${e.message}. Make sure the FastAPI backend is running.`)
    } finally { setLoading(false) }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-mark">CC</div>
        <div><div className="brand">COMPLAINT INTELLIGENCE</div><div className="subbrand">Consumer complaint classification</div></div>
        <div className={`status ${apiOnline ? 'online' : ''}`}><span />{apiOnline ? 'API Connected' : 'API Disconnected'}</div>
      </header>

      <section className="hero">
        <div className="eyebrow">AI-POWERED ANALYSIS</div>
        <h1>Understand every<br/><em>complaint.</em></h1>
        <p>Classify consumer complaints with four deep-learning models and compare their predictions instantly.</p>
      </section>

      <section className="workspace">
        <div className="section-head"><div><span className="step">01</span><h2>Complaint details</h2></div><span className="muted">TEXT CLASSIFICATION</span></div>
        <label htmlFor="complaint">COMPLAINT NARRATIVE</label>
        <textarea id="complaint" value={complaint} onChange={e => setComplaint(e.target.value)}
          placeholder="Describe the consumer complaint here..." />
        <div className="input-foot"><span>{complaint.length} characters</span><span>English text works best</span></div>

        <div className="section-head model-head"><div><span className="step">02</span><h2>Choose a model</h2></div><span className="muted">4 MODELS AVAILABLE</span></div>
        <div className="model-grid">
          {models.map(m => <button key={m.id} className={`model ${selected === m.id ? 'chosen' : ''}`} onClick={() => setSelected(m.id)}>
            <span className="radio">{selected === m.id ? '●' : '○'}</span><strong>{m.name}</strong><small>{m.id === 'transformer' ? 'Transformer architecture' : 'Recurrent neural network'}</small>
          </button>)}
          {!models.length && ['SimpleRNN','LSTM','GRU','Transformer'].map(id => <div className="model unavailable" key={id}><strong>{id}</strong><small>Waiting for API connection</small></div>)}
        </div>

        <div className="actions">
          <button className="primary" disabled={loading} onClick={() => submit('/predict')}>{loading ? 'Analyzing…' : 'Classify complaint'}</button>
          <button className="secondary" disabled={loading} onClick={() => submit('/predict/all')}>Compare all models</button>
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {result && <section className="result-card">
        <div className="result-label">PREDICTION RESULT</div><h2>{result.predicted_category.replaceAll('_',' ')}</h2>
        <p>{result.model} · {(result.confidence * 100).toFixed(2)}% confidence</p>
        <div className="bar"><span style={{width:`${result.confidence*100}%`}} /></div>
        <ProbabilityList probabilities={result.probabilities}/>
      </section>}

      {allResults && <section className="result-card">
        <div className="result-label">MODEL COMPARISON</div><h2>Four perspectives</h2>
        <div className="compare-list">{allResults.predictions.map(item => <div className="compare-row" key={item.model_key}>
          <div><strong>{item.model}</strong><small>{item.predicted_category.replaceAll('_',' ')}</small></div>
          <b>{(item.confidence*100).toFixed(1)}%</b>
          <div className="mini-bar"><span style={{width:`${item.confidence*100}%`}}/></div>
        </div>)}</div>
        <p className="vote">Majority vote: <strong>{allResults.majority_vote.predicted_category.replaceAll('_',' ')}</strong> ({allResults.majority_vote.votes}/4 models)</p>
      </section>}
      <footer>LOCAL INFERENCE · YOUR COMPLAINT TEXT STAYS ON THIS MACHINE</footer>
    </main>
  )
}

function ProbabilityList({ probabilities }) {
  if (!probabilities) return null
  return <div className="probabilities">{Object.entries(probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value]) =>
    <div key={name}><span>{name.replaceAll('_',' ')}</span><b>{(value*100).toFixed(1)}%</b></div>)}</div>
}
