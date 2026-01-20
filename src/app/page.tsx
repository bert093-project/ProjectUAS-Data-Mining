'use client';

import React, { useState } from 'react';
import DatasetPanel from './components/DatasetPanel';
import ProcessPanel from './components/ProcessPanel';
import PerformancePanel from './components/PerformancePanel';
import RecommendedList from './components/RecommendedList';
import type { DataRow } from './naive-bayes/model';
import { detectColumnTypes, inferCandidateTarget } from './utils/dataUtils';

/**
 * Note: contoh dataset yang diunggah tersedia pada file upload
 * (converted JSON/CSV) — lihat: :contentReference[oaicite:1]{index=1}
 */

type SplitResponse = {
  mode: 'split';
  counts: { total: number; train: number; test: number };
  model: { classes: string[]; featureColumns: string[]; targetColumn: string };
  eval: {
    accuracy: number;
    confusionMatrix: number[][];
    precisionPerClass: Record<string, number>;
    recallPerClass: Record<string, number>;
    f1PerClass: Record<string, number>;
    macroPrecision?: number;
    macroRecall?: number;
    macroF1?: number;
  };
  scoredItems: Array<DataRow & { predictedLabel?: string; probabilities?: Record<string, number> }>;
};

type ApiResponse = SplitResponse | { error?: string };

/** ScoredItem used for RecommendedList: require predictedLabel & probabilities */
type ScoredItem = DataRow & {
  predictedLabel: string;
  probabilities: Record<string, number>;
};

export default function Page() {
  // data + meta
  const [data, setData] = useState<DataRow[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [columnTypes, setColumnTypes] = useState<Record<string, 'numeric' | 'categorical'>>({});
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<'dataset' | 'process' | 'performance'>('dataset');

  // evaluation / result
  const [loading, setLoading] = useState<boolean>(false);
  const [apiResult, setApiResult] = useState<ApiResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // recommendation state — use ScoredItem type
  const [budget, setBudget] = useState<number | null>(null);
  // default budgetLabel is price-class selection (cheap/mid/premium)
  const [budgetLabel, setBudgetLabel] = useState<string>('mid');
  const [topN, setTopN] = useState<number>(5);
  const [recommendedItems, setRecommendedItems] = useState<ScoredItem[]>([]);

  // callbacks from DatasetPanel
  function handleDataLoaded(d: DataRow[]) {
    setData(d);
    const cols = d.length ? Object.keys(d[0]) : [];
    setColumns(cols);
    const types = detectColumnTypes(d);
    setColumnTypes(types);
    const suggested = inferCandidateTarget(d);
    if (suggested) {
      setTargetColumn(suggested);
      setFeatureColumns(cols.filter(c => c !== suggested));
    } else {
      setTargetColumn(null);
      setFeatureColumns(cols);
    }
    setActiveTab('process');
  }

  function handleColumnsDetected(cols: string[]) {
    setColumns(cols);
  }
  function handleColumnTypes(types: Record<string, 'numeric' | 'categorical'>) {
    setColumnTypes(types);
  }

  // Helper parse number safely
  function parseNumberSafe(v: unknown): number {
    if (v === null || v === undefined) return NaN;
    if (typeof v === 'number') return v;
    const s = String(v).replace(/[^\d.-]/g, '');
    const n = parseFloat(s);
    return Number.isFinite(n) ? n : NaN;
  }

  // Price-to-label mapping (user requested premium if price > 17,000,000)
  function priceToLabel(price: number): 'cheap' | 'mid' | 'premium' {
    if (!Number.isFinite(price)) return 'mid';
    if (price < 5_000_000) return 'cheap';
    if (price > 17_000_000) return 'premium';
    return 'mid';
  }

  // API call to run evaluation (split only)
// ganti seluruh function runEval lama dengan ini
  async function runEval(cfg: { mode: 'split'; trainPercent: number }) {
    if (!data) {
      alert('No dataset loaded');
      return;
    }
    if (!targetColumn) {
      alert('Select target column in Process panel');
      return;
    }
    if (featureColumns.length === 0) {
      alert('Select at least one feature column in Process panel');
      return;
    }

    setLoading(true);
    setApiResult(null);
    setApiError(null);
    setRecommendedItems([] as ScoredItem[]);

    try {
      // salin data supaya tidak mengubah state asli
      const dataCopy = (data || []).map((r) => ({ ...r }));

      // jika user memilih price_idr sebagai target, buat price_class otomatis
      let effectiveTarget = targetColumn;
      const numericPriceKeys = ['price_idr', 'price', 'price_id'];
      if (targetColumn === 'price_idr' || numericPriceKeys.includes(targetColumn || '')) {
        // thresholds: cheap <5M, mid 5M-17M, premium >17M
        dataCopy.forEach((row) => {
          const raw = row['price_idr'] ?? row['price'] ?? row['price_id'] ?? null;
          const n = (() => {
            if (raw === null || raw === undefined) return NaN;
            if (typeof raw === 'number') return raw;
            const s = String(raw).replace(/[^\d.-]/g, '');
            const v = parseFloat(s);
            return Number.isFinite(v) ? v : NaN;
          })();
          let cls = 'mid';
          if (!Number.isFinite(n)) cls = 'mid';
          else if (n < 5_000_000) cls = 'cheap';
          else if (n > 17_000_000) cls = 'premium';
          else cls = 'mid';
          // tulis kolom baru
          (row as any).price_class = cls;
        });
        effectiveTarget = 'price_class';

        // pastikan effectiveTarget bukan di featureColumns
        const featuresNoTarget = featureColumns.filter(c => c !== 'price_idr' && c !== 'price' && c !== 'price_id' && c !== effectiveTarget);
        // prepare payload
        const payload = {
          data: dataCopy,
          featureColumns: featuresNoTarget,
          targetColumn: effectiveTarget,
          eval: cfg
        };

        console.log('[runEval] using derived target price_class; payload sample:', payload.featureColumns, payload.targetColumn, { rows: dataCopy.length });

        const res = await fetch('/api/naive-bayes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const json = await res.json().catch(e => {
          console.error('[runEval] failed parse JSON', e);
          return null;
        });
        console.log('[runEval] response status:', res.status, res.statusText);
        console.log('[runEval] response json:', json);

        if (!res.ok) {
          const msg = json && json.error ? String(json.error) : `HTTP ${res.status}`;
          setApiError(msg);
          alert(`Evaluation failed: ${msg}`);
          setApiResult(null);
          return;
        }
        setApiResult(json as ApiResponse);
        // set budget label default
        setBudgetLabel('mid');
        setActiveTab('performance');
        return;
    }

    // jika bukan price_idr target -> perilaku lama, pastikan fitur tidak mengandung target
    const featuresNoTarget = featureColumns.filter(c => c !== targetColumn);
    const payload = { data: dataCopy, featureColumns: featuresNoTarget, targetColumn, eval: cfg };
    console.log('[runEval] payload:', { featureColumns: featuresNoTarget, targetColumn, trainPercent: cfg.trainPercent, rows: dataCopy.length });

    const res = await fetch('/api/naive-bayes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json().catch(e => {
      console.error('[runEval] failed parse JSON', e);
      return null;
    });
    console.log('[runEval] response status:', res.status, res.statusText);
    console.log('[runEval] response json:', json);

    if (!res.ok) {
      const msg = json && json.error ? String(json.error) : `HTTP ${res.status}`;
      setApiError(msg);
      alert(`Evaluation failed: ${msg}`);
      setApiResult(null);
      return;
    }
    setApiResult(json as ApiResponse);
    setActiveTab('performance');
  } catch (err) {
    console.error('[runEval] error:', err);
    const msg = err instanceof Error ? err.message : String(err);
    setApiError(msg);
    alert(`Evaluation error: ${msg}`);
  } finally {
    setLoading(false);
  }
}


  // recommendation helpers
  function recommendByLabel(label: string): ScoredItem[] {
    if (!apiResult) return [] as ScoredItem[];

    // If API scoredItems available with probabilities, normalize and use probabilities
    if ('scoredItems' in apiResult && Array.isArray((apiResult as any).scoredItems)) {
      const items = (apiResult as any).scoredItems as (DataRow & { predictedLabel?: string; probabilities?: Record<string, number> })[];

      // normalize to required ScoredItem type
      const normalized: ScoredItem[] = items.map((it) => {
        return {
          ...(it as DataRow),
          predictedLabel: it.predictedLabel ?? '',
          probabilities: it.probabilities ?? {}
        };
      });

      // If probabilities contain the requested label, sort by it
      const hasLabel = normalized.some(it => Object.prototype.hasOwnProperty.call(it.probabilities, label));
      if (hasLabel) {
        const sorted = normalized.slice().sort((a, b) => {
          const pa = a.probabilities?.[label] ?? 0;
          const pb = b.probabilities?.[label] ?? 0;
          return pb - pa;
        });
        return sorted.slice(0, Math.max(0, topN));
      }

      // Fallback: filter by price range using priceToLabel
      const fallback = normalized.filter(it => {
        const p = parseNumberSafe(it['price_idr'] ?? it['price'] ?? it['price_id'] ?? '');
        return priceToLabel(p) === (label as 'cheap' | 'mid' | 'premium');
      }).slice(0, Math.max(0, topN));
      if (fallback.length > 0) return fallback;
    }

    // If no scoredItems or not matching, fallback to raw `data` filtering by price
    if (data && data.length > 0) {
      const candidates = data.map(d => {
        const p = parseNumberSafe(d['price_idr'] ?? d['price'] ?? '');
        const scored: ScoredItem = {
          ...(d as DataRow),
          predictedLabel: priceToLabel(p),
          probabilities: {}
        };
        return scored;
      }).filter(it => it.predictedLabel === label).slice(0, Math.max(0, topN));
      return candidates;
    }

    return [] as ScoredItem[];
  }

  function onRecommendClick() {
    if (!apiResult && !data) return alert('Run evaluation first or upload data');
    let rec: ScoredItem[] = [];
    if (budget !== null && Number.isFinite(budget)) {
      const label = priceToLabel(budget);
      setBudgetLabel(label);
      rec = recommendByLabel(label);
    } else if (budgetLabel) {
      rec = recommendByLabel(budgetLabel);
    } else {
      alert('Set budget number or select a price class (cheap/mid/premium)');
      return;
    }
    setRecommendedItems(rec);
    const el = document.getElementById('recommendations');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  }

  // small UI helpers for showing eval results (split only)
  function renderEvalSection() {
    if (apiError) {
      return <div className="text-sm text-red-600">Error: {apiError}</div>;
    }
    if (!apiResult) return <div className="text-sm text-slate-600">No evaluation run yet.</div>;

    if ('mode' in apiResult && apiResult.mode === 'split') {
      const sp = apiResult as SplitResponse;
      const accuracyPct = ((sp.eval?.accuracy ?? 0) * 100);
      const macroF1Pct = ((sp.eval?.macroF1 ?? 0) * 100);
      return (
        <div>
          <div className="mb-2">Split evaluation</div>
          <div className="mb-2">Total: {sp.counts?.total ?? '?'} — Train: {sp.counts?.train ?? '?'} — Test: {sp.counts?.test ?? '?'}</div>
          <div className="mb-2">Accuracy: {accuracyPct.toFixed(2)}%</div>
          <div className="mb-2">Macro F1: {macroF1Pct.toFixed(2)}%</div>
        </div>
      );
    }

    return <div className="text-sm text-slate-600">Unknown result format</div>;
  }

  return (
    <main className="min-h-screen p-6 bg-slate-50">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Left panel (tabs) */}
        <aside className="col-span-1 bg-white rounded-2xl shadow p-4 sticky top-6">
          <h2 className="text-lg font-semibold mb-3">Pipeline</h2>

          <div className="flex flex-col gap-2 mb-4">
            <button className={`text-left px-3 py-2 rounded ${activeTab === 'dataset' ? 'bg-slate-100' : ''}`} onClick={() => setActiveTab('dataset')}>1. Dataset</button>
            <button className={`text-left px-3 py-2 rounded ${activeTab === 'process' ? 'bg-slate-100' : ''}`} onClick={() => setActiveTab('process')}>2. Process</button>
            <button className={`text-left px-3 py-2 rounded ${activeTab === 'performance' ? 'bg-slate-100' : ''}`} onClick={() => setActiveTab('performance')}>3. Performance</button>
          </div>

          <div className="border-t pt-4">
            <div className="text-sm text-slate-600 mb-2">Selected target</div>
            <div className="mb-3">
              <div className="px-3 py-2 border rounded bg-yellow-50 text-yellow-900">{targetColumn ?? '—'}</div>
            </div>

            <div className="text-sm text-slate-600 mb-2">Selected features ({featureColumns.length})</div>
            <div className="space-y-1">
              {featureColumns.map((c) => (
                <div key={c} className="px-2 py-1 border rounded text-sm bg-blue-50 text-blue-800">{c}</div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main content */}
        <section className="col-span-3">
          <div className="bg-white rounded-2xl shadow p-4">
            {/* Tab content */}
            {activeTab === 'dataset' && (
              <DatasetPanel
                data={data}
                onDataLoaded={(d) => handleDataLoaded(d)}
                onColumnsDetected={(cols) => handleColumnsDetected(cols)}
                onColumnTypes={(types) => handleColumnTypes(types)}
              />
            )}

            {activeTab === 'process' && (
              <ProcessPanel
                columns={columns}
                featureColumns={featureColumns}
                targetColumn={targetColumn}
                setFeatureColumns={(cols) => setFeatureColumns(cols)}
                setTargetColumn={(col) => {
                  setTargetColumn(col);
                  setFeatureColumns(prev => prev.filter(c => c !== col));
                }}
                columnTypes={columnTypes}
              />
            )}

            {activeTab === 'performance' && (
              <div>
                <PerformancePanel
                  dataCount={data ? data.length : 0}
                  featureColumns={featureColumns}
                  targetColumn={targetColumn}
                  onRunEval={(cfg) => runEval(cfg)}
                />

                <div className="mt-4">
                  <div className="flex items-center gap-2">
                    <div className="text-sm font-medium">Evaluation results</div>
                    {loading && <div className="text-xs text-slate-500">Running...</div>}
                  </div>
                  <div className="mt-3 p-3 border rounded">
                    {renderEvalSection()}
                  </div>

                  <div className="mt-4">
                    <h4 className="font-medium mb-2">Recommendation</h4>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-2 mb-3">
                      <div>
                        <label className="text-xs">Budget (IDR)</label>
                        <input type="number" value={budget ?? ''} onChange={(e) => setBudget(Number(e.target.value) || null)} className="w-full border rounded px-2 py-1" placeholder="e.g. 10000000" />
                      </div>
                      <div>
                        <label className="text-xs">Or select price class</label>
                        <select value={budgetLabel} onChange={(e) => setBudgetLabel(e.target.value)} className="w-full border rounded px-2 py-1">
                          <option value="cheap">cheap (&lt; 5.000.000)</option>
                          <option value="mid">mid (5.000.000 - 17.000.000)</option>
                          <option value="premium">premium (&gt; 17.000.000)</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-xs">Top N</label>
                        <input type="number" value={topN} onChange={(e) => setTopN(Math.max(1, Number(e.target.value) || 5))} className="w-full border rounded px-2 py-1" />
                      </div>
                      <div className="flex items-end">
                        <button className="px-4 py-2 bg-green-600 text-white rounded" onClick={onRecommendClick}>Recommend</button>
                      </div>
                    </div>

                    <div id="recommendations" className="mt-3">
                      <RecommendedList items={recommendedItems} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Raw scored items preview (optional) */}
          {apiResult && typeof apiResult === 'object' && Array.isArray((apiResult as any).scoredItems) && (
            <div className="mt-6 bg-white rounded-2xl shadow p-4">
              <h3 className="font-semibold mb-2">All scored items (preview)</h3>
              <div className="overflow-auto">
                <table className="table-auto w-full text-sm">
                  <thead>
                    <tr>
                      {columns.concat(['predictedLabel']).map(c => <th key={c} className="border px-2 py-1 text-left">{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {((apiResult as any).scoredItems as any[]).slice(0, 20).map((row, i) => (
                      <tr key={i}>
                        {columns.map(col => <td key={col} className="border px-2 py-1">{String(row[col] ?? '')}</td>)}
                        <td className="border px-2 py-1">{String(row.predictedLabel ?? '')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="text-xs text-slate-600 mt-2">Showing up to 20 rows</div>
              </div>
            </div>
          )}

          {/* If there's an API error, show it */}
          {apiError && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-400 text-red-800 p-3 rounded">
              <div className="font-medium">API Error</div>
              <div className="text-sm">{apiError}</div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}