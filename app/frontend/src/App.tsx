import { useState, useEffect, useRef, useCallback } from 'react'

// ============================================================================
// TYPES
// ============================================================================
interface Strategy { id: number; name: string; filename: string }
interface Asset { name: string; timeframes: string[] }
interface Progress { trial: number; total: number; strategy: string; asset: string; best_score: number; status: string; eta: number | null }
interface SystemStatus { cpu: number; ram: number; is_running: boolean; progress: Progress }
interface Chart { name: string; path: string; strategy: string; asset: string; score: number }
interface Summary { name: string; path: string; full_path: string }
interface ResultsTree { strategies: Record<string, { timeframes: Record<string, { charts: any[]; csv: any[] }>; total: number }>; total_files: number }
interface AnalysisResult { parameter: string; correlation: number; correlation_strength: string; optimal_range: { min: number; max: number; mean: number }; significance?: { significant: boolean; p_value: number }; n_samples: number }
interface NoiseAnalysis { noise_level: string; distribution: { mean: number; std: number; cv: number }; outliers: { count: number; percentage: number } }

const API_BASE = '/api'

// ============================================================================
// UTILITIES
// ============================================================================
const cn = (...classes: (string | boolean | undefined)[]) => classes.filter(Boolean).join(' ')
const formatPercent = (n: number) => `${n.toFixed(1)}%`
const formatNumber = (n: number) => n.toLocaleString('en-US', { maximumFractionDigits: 2 })
const formatEta = (seconds: number | null) => {
  if (!seconds) return '--:--'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const getScoreColor = (score: number) => {
  if (score >= 10) return 'text-emerald-400'
  if (score >= 5) return 'text-green-400'
  if (score >= 0) return 'text-yellow-400'
  return 'text-red-400'
}

const getCorrelationColor = (strength: string) => {
  switch (strength) {
    case 'strong': return 'text-emerald-400'
    case 'moderate': return 'text-yellow-400'
    case 'weak': return 'text-orange-400'
    default: return 'text-gray-400'
  }
}

const getNoiseColor = (level: string) => {
  switch (level) {
    case 'very_low': return 'text-emerald-400'
    case 'low': return 'text-green-400'
    case 'moderate': return 'text-yellow-400'
    case 'high': return 'text-orange-400'
    case 'very_high': return 'text-red-400'
    default: return 'text-gray-400'
  }
}

// ============================================================================
// MAIN APP
// ============================================================================
export default function App() {
  // Navigation
  const [view, setView] = useState<'dashboard' | 'results' | 'charts' | 'analysis'>('dashboard')
  
  // Data
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [assets, setAssets] = useState<Asset[]>([])
  const [charts, setCharts] = useState<Chart[]>([])
  const [summaries, setSummaries] = useState<Summary[]>([])
  const [resultsTree, setResultsTree] = useState<ResultsTree | null>(null)
  
  // Config
  const [selectedStrategies, setSelectedStrategies] = useState<number[]>([])
  const [selectedAsset, setSelectedAsset] = useState('')
  const [selectedTimeframe, setSelectedTimeframe] = useState('1m')
  const [nTrials, setNTrials] = useState(100)
  
  // Status
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  
  // Views
  const [selectedChart, setSelectedChart] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [analysisResults, setAnalysisResults] = useState<Record<string, AnalysisResult> | null>(null)
  const [noiseAnalysis, setNoiseAnalysis] = useState<NoiseAnalysis | null>(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  
  const logsEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const statusWsRef = useRef<WebSocket | null>(null)

  // ============================================================================
  // FETCH INITIAL DATA
  // ============================================================================
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/strategies`).then(r => r.json()),
      fetch(`${API_BASE}/assets`).then(r => r.json()),
    ]).then(([strats, assts]) => {
      setStrategies(strats)
      setAssets(assts)
      if (assts.length > 0) {
        setSelectedAsset(assts[0].name)
        if (assts[0].timeframes?.length > 0) {
          setSelectedTimeframe(assts[0].timeframes[0])
        }
      }
    }).catch(console.error)
  }, [])

  // ============================================================================
  // WEBSOCKET FOR STATUS
  // ============================================================================
  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/status`)
      
      ws.onopen = () => {
        console.log('Status WS connected')
        setWsConnected(true)
      }
      
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          setStatus(data)
        } catch {}
      }
      
      ws.onclose = () => {
        console.log('Status WS disconnected')
        setWsConnected(false)
        setTimeout(connect, 2000)
      }
      
      ws.onerror = () => {
        ws.close()
      }
      
      statusWsRef.current = ws
    }
    
    connect()
    return () => statusWsRef.current?.close()
  }, [])

  // ============================================================================
  // WEBSOCKET FOR LOGS
  // ============================================================================
  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/logs`)
      
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          if (data.logs?.length) {
            setLogs(prev => [...prev, ...data.logs].slice(-1000))
          }
        } catch {}
      }
      
      ws.onclose = () => {
        setTimeout(connect, 2000)
      }
      
      wsRef.current = ws
    }
    
    connect()
    return () => wsRef.current?.close()
  }, [])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Load results when switching views
  useEffect(() => {
    if (view === 'results') {
      fetch(`${API_BASE}/results/tree`).then(r => r.json()).then(setResultsTree)
      fetch(`${API_BASE}/results/summaries`).then(r => r.json()).then(setSummaries)
    }
    if (view === 'charts') {
      fetch(`${API_BASE}/results/charts?limit=200`).then(r => r.json()).then(setCharts)
    }
  }, [view])

  // ============================================================================
  // HANDLERS
  // ============================================================================
  const handleStart = async () => {
    try {
      const res = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset: selectedAsset,
          timeframe: selectedTimeframe,
          n_trials: nTrials,
          strategy_ids: selectedStrategies.length ? selectedStrategies : strategies.map(s => s.id)
        })
      })
      if (res.ok) {
        setLogs([])
      } else {
        const err = await res.json()
        alert(`Error: ${err.detail || 'Unknown error'}`)
      }
    } catch (e) {
      console.error(e)
    }
  }

  const handleStop = async () => {
    await fetch(`${API_BASE}/stop`, { method: 'POST' })
  }

  const toggleStrategy = (id: number) => {
    setSelectedStrategies(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    )
  }

  const loadAnalysis = async (filePath: string) => {
    setAnalysisLoading(true)
    setSelectedFile(filePath)
    setAnalysisResults(null)
    setNoiseAnalysis(null)
    
    try {
      const [paramsRes, noiseRes] = await Promise.all([
        fetch(`${API_BASE}/analysis/parameters`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_path: filePath, target_metric: 'SCORE' })
        }).then(r => r.json()),
        fetch(`${API_BASE}/analysis/noise`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_path: filePath, target_metric: 'SCORE' })
        }).then(r => r.json())
      ])
      
      setAnalysisResults(paramsRes.impacts)
      setNoiseAnalysis(noiseRes)
    } catch (e) {
      console.error(e)
    } finally {
      setAnalysisLoading(false)
    }
  }

  const currentAsset = assets.find(a => a.name === selectedAsset)
  const isRunning = status?.is_running || false
  const progress = status?.progress

  // ============================================================================
  // RENDER
  // ============================================================================
  return (
    <div className="min-h-screen bg-[#080b10] text-white">
      {/* Header */}
      <header className="border-b border-cyan-900/30 bg-[#0a0e14]/90 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-[1920px] mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-600 flex items-center justify-center text-xl font-black shadow-lg shadow-cyan-500/20">
                M
              </div>
              <div>
                <h1 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  MODELOX
                </h1>
                <p className="text-[10px] text-cyan-400/70 tracking-[0.2em] font-medium">QUANT STATION v3.0</p>
              </div>
            </div>
          </div>
          
          {/* Navigation */}
          <nav className="flex gap-1 bg-[#0d1117] rounded-xl p-1 border border-gray-800">
            {(['dashboard', 'results', 'charts', 'analysis'] as const).map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={cn(
                  "px-5 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  view === v 
                    ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 shadow-lg shadow-cyan-500/10" 
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                )}
              >
                {v === 'dashboard' ? '⚡ Dashboard' : 
                 v === 'results' ? '📊 Results' :
                 v === 'charts' ? '📈 Charts' : '🔬 Analysis'}
              </button>
            ))}
          </nav>
          
          {/* Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-4 text-sm bg-[#0d1117] rounded-xl px-4 py-2 border border-gray-800">
              <div className="flex items-center gap-2">
                <span className="text-gray-500 text-xs">CPU</span>
                <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className={cn("h-full transition-all", (status?.cpu || 0) > 80 ? "bg-red-500" : "bg-cyan-500")}
                    style={{ width: `${status?.cpu || 0}%` }}
                  />
                </div>
                <span className={cn("font-mono text-xs", (status?.cpu || 0) > 80 ? "text-red-400" : "text-cyan-400")}>
                  {formatPercent(status?.cpu || 0)}
                </span>
              </div>
              <div className="w-px h-4 bg-gray-700" />
              <div className="flex items-center gap-2">
                <span className="text-gray-500 text-xs">RAM</span>
                <span className={cn("font-mono text-xs", (status?.ram || 0) > 80 ? "text-red-400" : "text-green-400")}>
                  {formatPercent(status?.ram || 0)}
                </span>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <div className={cn(
                "w-2 h-2 rounded-full",
                wsConnected ? "bg-green-500" : "bg-red-500"
              )} />
              <div className={cn(
                "w-3 h-3 rounded-full transition-all",
                isRunning ? "bg-green-500 animate-pulse shadow-lg shadow-green-500/50" : "bg-gray-600"
              )} />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1920px] mx-auto p-6">
        {/* ================================================================
            DASHBOARD VIEW
        ================================================================ */}
        {view === 'dashboard' && (
          <div className="grid grid-cols-12 gap-6">
            {/* Control Panel */}
            <div className="col-span-4 space-y-4">
              {/* Config Card */}
              <div className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-6 shadow-xl">
                <h2 className="text-sm font-bold text-cyan-400 mb-5 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  OPTIMIZATION CONFIG
                </h2>
                
                {/* Asset Selector */}
                <div className="space-y-4">
                  <label className="block">
                    <span className="text-xs text-gray-500 mb-2 block font-medium">ASSET</span>
                    <select
                      value={selectedAsset}
                      onChange={e => setSelectedAsset(e.target.value)}
                      className="w-full bg-[#080b10] border border-gray-700 rounded-xl px-4 py-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 outline-none transition-all"
                    >
                      {assets.map(a => (
                        <option key={a.name} value={a.name}>{a.name}</option>
                      ))}
                    </select>
                  </label>
                  
                  <label className="block">
                    <span className="text-xs text-gray-500 mb-2 block font-medium">TIMEFRAME</span>
                    <select
                      value={selectedTimeframe}
                      onChange={e => setSelectedTimeframe(e.target.value)}
                      className="w-full bg-[#080b10] border border-gray-700 rounded-xl px-4 py-3 text-white focus:border-cyan-500 outline-none transition-all"
                    >
                      {currentAsset?.timeframes.map(tf => (
                        <option key={tf} value={tf}>{tf}</option>
                      ))}
                    </select>
                  </label>
                  
                  <label className="block">
                    <span className="text-xs text-gray-500 mb-2 block font-medium">TRIALS</span>
                    <input
                      type="number"
                      value={nTrials}
                      onChange={e => setNTrials(Math.max(1, parseInt(e.target.value) || 1))}
                      className="w-full bg-[#080b10] border border-gray-700 rounded-xl px-4 py-3 text-white focus:border-cyan-500 outline-none font-mono transition-all"
                    />
                  </label>
                </div>

                {/* Strategies */}
                <div className="mt-5">
                  <span className="text-xs text-gray-500 mb-3 block font-medium">
                    STRATEGIES ({selectedStrategies.length || 'ALL'})
                  </span>
                  <div className="max-h-48 overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-gray-700">
                    {strategies.map(s => (
                      <button
                        key={s.id}
                        onClick={() => toggleStrategy(s.id)}
                        className={cn(
                          "w-full text-left px-4 py-2.5 rounded-xl text-sm transition-all flex items-center gap-3",
                          selectedStrategies.includes(s.id)
                            ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/10"
                            : "bg-[#080b10] text-gray-400 hover:text-white border border-gray-800 hover:border-gray-600"
                        )}
                      >
                        <span className={cn(
                          "w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold",
                          selectedStrategies.includes(s.id) ? "bg-cyan-500/30" : "bg-gray-800"
                        )}>
                          {selectedStrategies.includes(s.id) ? '✓' : s.id}
                        </span>
                        <span className="truncate font-medium">{s.name}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="mt-6 flex gap-3">
                  <button
                    onClick={handleStart}
                    disabled={isRunning}
                    className={cn(
                      "flex-1 py-4 rounded-xl font-bold text-sm transition-all",
                      isRunning 
                        ? "bg-gray-800 text-gray-500 cursor-not-allowed"
                        : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:shadow-xl hover:shadow-cyan-500/30 active:scale-[0.98]"
                    )}
                  >
                    {isRunning ? '⏳ RUNNING...' : '▶ START OPTIMIZATION'}
                  </button>
                  {isRunning && (
                    <button
                      onClick={handleStop}
                      className="px-5 py-4 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-all font-bold"
                    >
                      ■ STOP
                    </button>
                  )}
                </div>
              </div>

              {/* Progress Card */}
              {isRunning && progress && (
                <div className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-green-900/30 p-6 shadow-xl shadow-green-500/5">
                  <h2 className="text-sm font-bold text-green-400 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    OPTIMIZATION PROGRESS
                  </h2>
                  
                  <div className="space-y-5">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-white font-mono font-bold">{progress.trial} / {progress.total}</span>
                      </div>
                      <div className="h-3 bg-[#080b10] rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-500 rounded-full"
                          style={{ width: `${(progress.trial / progress.total) * 100}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                        <span className="text-gray-500 block text-xs mb-1">Best Score</span>
                        <span className={cn("text-3xl font-black", getScoreColor(progress.best_score))}>
                          {progress.best_score.toFixed(2)}
                        </span>
                      </div>
                      <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                        <span className="text-gray-500 block text-xs mb-1">ETA</span>
                        <span className="text-2xl font-bold text-white font-mono">
                          {formatEta(progress.eta)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Terminal */}
            <div className="col-span-8">
              <div className="bg-[#0a0e14] rounded-2xl border border-gray-800 h-[calc(100vh-180px)] flex flex-col shadow-2xl">
                <div className="px-5 py-3 border-b border-gray-800 flex items-center justify-between bg-[#0d1117] rounded-t-2xl">
                  <div className="flex items-center gap-3">
                    <div className="flex gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500" />
                      <div className="w-3 h-3 rounded-full bg-green-500" />
                    </div>
                    <span className="text-sm text-gray-500 font-medium">Terminal Output</span>
                    {isRunning && (
                      <span className="text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded-lg animate-pulse">
                        LIVE
                      </span>
                    )}
                  </div>
                  <button 
                    onClick={() => setLogs([])}
                    className="text-xs text-gray-500 hover:text-white px-3 py-1 rounded-lg hover:bg-white/5 transition-all"
                  >
                    Clear
                  </button>
                </div>
                
                <div className="flex-1 overflow-y-auto p-5 font-mono text-sm leading-relaxed">
                  {logs.length === 0 ? (
                    <div className="text-gray-600 text-center mt-32">
                      <div className="text-6xl mb-4">⚡</div>
                      <p className="text-xl font-bold text-gray-500">Ready to Optimize</p>
                      <p className="text-sm text-gray-600 mt-2">Configure parameters and start optimization</p>
                    </div>
                  ) : (
                    logs.map((log, i) => (
                      <div 
                        key={i} 
                        className={cn(
                          "py-0.5 break-all",
                          log.includes('ERROR') || log.includes('Error') ? 'text-red-400' :
                          log.includes('WARNING') ? 'text-yellow-400' :
                          log.includes('TRIAL') || log.includes('Trial') ? 'text-cyan-400' :
                          log.includes('BEST') || log.includes('Best') || log.includes('SCORE') ? 'text-green-400' :
                          log.includes('[FINISHED]') ? 'text-purple-400 font-bold' :
                          'text-gray-400'
                        )}
                      >
                        {log}
                      </div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================
            RESULTS VIEW
        ================================================================ */}
        {view === 'results' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">📊 Results Explorer</h2>
              {resultsTree && (
                <span className="text-gray-500">{resultsTree.total_files} files</span>
              )}
            </div>
            
            {resultsTree && Object.keys(resultsTree.strategies).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(resultsTree.strategies).map(([name, data]) => (
                  <div key={name} className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-6 hover:border-cyan-500/30 transition-all group">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="font-bold text-white truncate group-hover:text-cyan-400 transition-colors">{name}</h3>
                        <p className="text-sm text-gray-500">{data.total} files</p>
                      </div>
                      <span className="text-3xl">📈</span>
                    </div>
                    
                    <div className="space-y-2">
                      {Object.entries(data.timeframes).map(([tf, files]) => (
                        <div key={tf} className="text-sm bg-[#080b10] rounded-xl p-3 border border-gray-800">
                          <div className="flex items-center gap-2 text-gray-400">
                            <span className="text-cyan-400 font-bold">{tf}</span>
                            <span className="text-gray-600">•</span>
                            <span className="text-green-400">{files.charts.length} charts</span>
                            <span className="text-gray-600">•</span>
                            <span className="text-blue-400">{files.csv.length} CSV</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-20 text-gray-500">
                <div className="text-6xl mb-4">📭</div>
                <p className="text-xl">No results yet</p>
                <p className="text-sm mt-2">Run an optimization to generate results</p>
              </div>
            )}
          </div>
        )}

        {/* ================================================================
            CHARTS VIEW
        ================================================================ */}
        {view === 'charts' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">📈 Interactive Charts</h2>
              <span className="text-gray-500">{charts.length} charts</span>
            </div>
            
            {selectedChart ? (
              <div className="space-y-4">
                <button 
                  onClick={() => setSelectedChart(null)}
                  className="text-cyan-400 hover:text-white flex items-center gap-2 font-medium"
                >
                  ← Back to list
                </button>
                <iframe 
                  src={selectedChart}
                  className="w-full h-[calc(100vh-280px)] rounded-2xl border border-gray-800 bg-white"
                />
              </div>
            ) : charts.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {charts.map((chart, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedChart(chart.path)}
                    className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-5 text-left hover:border-cyan-500/30 hover:shadow-xl hover:shadow-cyan-500/10 transition-all group"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className={cn("text-2xl font-black", getScoreColor(chart.score))}>
                        {chart.score.toFixed(2)}
                      </span>
                      <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded-lg">{chart.asset}</span>
                    </div>
                    <p className="text-sm text-gray-400 truncate group-hover:text-white transition-colors font-medium">
                      {chart.name}
                    </p>
                    <p className="text-xs text-gray-600 mt-2 truncate">
                      {chart.strategy}
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-20 text-gray-500">
                <div className="text-6xl mb-4">📊</div>
                <p className="text-xl">No charts available</p>
              </div>
            )}
          </div>
        )}

        {/* ================================================================
            ANALYSIS VIEW
        ================================================================ */}
        {view === 'analysis' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">🔬 Parameter Analysis</h2>
            </div>
            
            {/* File Selector */}
            <div className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-6">
              <h3 className="text-sm font-bold text-cyan-400 mb-4">SELECT RESULTS FILE</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {summaries.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => loadAnalysis(s.full_path)}
                    className={cn(
                      "text-left px-4 py-3 rounded-xl text-sm transition-all border",
                      selectedFile === s.full_path
                        ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/30"
                        : "bg-[#080b10] text-gray-400 hover:text-white border-gray-800 hover:border-gray-600"
                    )}
                  >
                    <div className="font-medium truncate">{s.name}</div>
                    <div className="text-xs text-gray-600 truncate mt-1">{s.path}</div>
                  </button>
                ))}
              </div>
            </div>
            
            {/* Analysis Results */}
            {analysisLoading && (
              <div className="text-center py-20">
                <div className="text-4xl animate-spin mb-4">⏳</div>
                <p className="text-gray-400">Analyzing parameters...</p>
              </div>
            )}
            
            {noiseAnalysis && !analysisLoading && (
              <div className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-6">
                <h3 className="text-sm font-bold text-purple-400 mb-4">📊 NOISE ANALYSIS</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                    <span className="text-xs text-gray-500 block mb-1">Noise Level</span>
                    <span className={cn("text-xl font-bold", getNoiseColor(noiseAnalysis.noise_level))}>
                      {noiseAnalysis.noise_level.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                    <span className="text-xs text-gray-500 block mb-1">CV (σ/μ)</span>
                    <span className="text-xl font-bold text-white">
                      {(noiseAnalysis.distribution.cv * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                    <span className="text-xs text-gray-500 block mb-1">Outliers</span>
                    <span className="text-xl font-bold text-orange-400">
                      {noiseAnalysis.outliers.count} ({noiseAnalysis.outliers.percentage.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                    <span className="text-xs text-gray-500 block mb-1">Mean ± Std</span>
                    <span className="text-xl font-bold text-white">
                      {noiseAnalysis.distribution.mean.toFixed(2)} ± {noiseAnalysis.distribution.std.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            {analysisResults && !analysisLoading && (
              <div className="bg-gradient-to-br from-[#0d1117] to-[#111920] rounded-2xl border border-cyan-900/20 p-6">
                <h3 className="text-sm font-bold text-green-400 mb-4">📈 PARAMETER IMPACTS (sorted by correlation)</h3>
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                  {Object.entries(analysisResults).map(([param, data]) => {
                    if ('error' in data) return null
                    const result = data as AnalysisResult
                    return (
                      <div key={param} className="bg-[#080b10] rounded-xl p-4 border border-gray-800">
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-bold text-white">{param}</span>
                          <div className="flex items-center gap-3">
                            <span className={cn("text-sm font-mono", getCorrelationColor(result.correlation_strength))}>
                              r = {result.correlation.toFixed(3)}
                            </span>
                            <span className={cn(
                              "text-xs px-2 py-1 rounded-lg",
                              result.correlation_strength === 'strong' ? "bg-emerald-500/20 text-emerald-400" :
                              result.correlation_strength === 'moderate' ? "bg-yellow-500/20 text-yellow-400" :
                              "bg-gray-800 text-gray-400"
                            )}>
                              {result.correlation_strength}
                            </span>
                            {result.significance?.significant && (
                              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded-lg">
                                p&lt;0.05
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500 text-xs">Optimal Range</span>
                            <div className="text-cyan-400 font-mono">
                              [{result.optimal_range.min.toFixed(1)}, {result.optimal_range.max.toFixed(1)}]
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-500 text-xs">Optimal Mean</span>
                            <div className="text-white font-mono">{result.optimal_range.mean.toFixed(2)}</div>
                          </div>
                          <div>
                            <span className="text-gray-500 text-xs">Samples</span>
                            <div className="text-white font-mono">{result.n_samples}</div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
            
            {!selectedFile && !analysisLoading && (
              <div className="text-center py-20 text-gray-500">
                <div className="text-6xl mb-4">🔬</div>
                <p className="text-xl">Select a results file to analyze</p>
                <p className="text-sm mt-2">Analysis includes parameter correlations, optimal ranges, and noise detection</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
