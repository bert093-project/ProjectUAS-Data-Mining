'use client';

import React, { useState } from 'react';
import DatasetPanel from './components/DatasetPanel';
import ProcessPanel from './components/ProcessPanel';
import PerformancePanel from './components/PerformancePanel';
import RecommendedList from './components/RecommendedList';
import type { DataRow } from './naive-bayes/model';
import { detectColumnTypes, inferCandidateTarget } from './utils/dataUtils';

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

  // recommendation state
  const [budget, setBudget] = useState<number | null>(null);
  const [budgetLabel, setBudgetLabel] = useState<string>('');
  const [topN, setTopN] = useState<number>(5);
  const [recommendedItems, setRecommendedItems] = useState<(DataRow & { predictedLabel?: string; probabilities?: Record<string, number> })[]>([]);

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

  // API call to run evaluation (split only)
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
    setRecommendedItems([]);

    try {
      const payload = {
        data,
        featureColumns,
        targetColumn,
        eval: cfg
      };
      console.log('[runEval] payload:', { featureColumns, targetColumn, trainPercent: cfg.trainPercent, rows: (data || []).length });

      const res = await fetch('/api/naive-bayes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      console.log('[runEval] response status:', res.status, res.statusText);

      const json = await res.json().catch((e) => {
        console.error('[runEval] failed to parse JSON:', e);
        return null;
      });

      console.log('[runEval] response json:', json);

      if (!res.ok) {
        const msg = (json && (json as any).error) ? String((json as any).error) : `HTTP ${res.status}`;
        setApiError(msg);
        alert(`Evaluation failed: ${msg}`);
        setApiResult(null);
        return;
      }

      setApiResult(json as ApiResponse);
      setApiError(null);

      const classes = json && typeof json === 'object' && 'model' in json && Array.isArray((json as any).model.classes)
        ? (json as any).model.classes as string[]
        : [];

      if (classes.length > 0) setBudgetLabel(classes[0]);
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
  function priceToLabel(price: number): string {
    if (!Number.isFinite(price)) return budgetLabel || '';
    if (price < 5_000_000) return 'cheap';
    if (price < 12_000_000) return 'mid';
    return 'premium';
  }

  function recommendByLabel(label: string) {
    if (!apiResult) return [];
    if (!('scoredItems' in apiResult) || !Array.isArray((apiResult as any).scoredItems)) return [];
    const items = (apiResult as any).scoredItems as (DataRow & { predictedLabel?: string; probabilities?: Record<string, number> })[];
    const sorted = items.slice().sort((a, b) => {
      const pa = a.probabilities?.[label] ?? 0;
      const pb = b.probabilities?.[label] ?? 0;
      return pb - pa;
    });
    return sorted.slice(0, Math.max(0, topN));
  }

  function onRecommendClick() {
    if (!apiResult) return alert('Run evaluation first');
    let rec: (DataRow & { predictedLabel?: string; probabilities?: Record<string, number> })[] = [];
    if (budget !== null && Number.isFinite(budget)) {
      const label = priceToLabel(budget);
      setBudgetLabel(label);
      rec = recommendByLabel(label);
    } else if (budgetLabel) {
      rec = recommendByLabel(budgetLabel);
    } else {
      alert('Set budget number or select a budget class');
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

    // Only split mode supported in current UI
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
                        <label className="text-xs">Or select class</label>
                        <select value={budgetLabel} onChange={(e) => setBudgetLabel(e.target.value)} className="w-full border rounded px-2 py-1">
                          {apiResult && typeof apiResult === 'object' && 'model' in apiResult && Array.isArray((apiResult as any).model.classes)
                            ? (apiResult as any).model.classes.map((c: string) => <option key={c} value={c}>{c}</option>)
                            : <option value=''>—</option>}
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