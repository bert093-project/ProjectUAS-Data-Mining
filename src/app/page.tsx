'use client';

import React, { useState } from 'react';
import * as XLSX from 'xlsx';

type DataRow = Record<string, string | number>;

type EvalResult = {
  accuracy: number;
  confusionMatrix: number[][];
  precisionPerClass: Record<string, number>;
  recallPerClass: Record<string, number>;
  f1PerClass: Record<string, number>;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
};

type ApiResponse = {
  model: {
    classes: string[];
    classPriors: Record<string, number>;
    features: Record<string, unknown>;
  };
  eval: EvalResult;
  counts: { total: number; train: number; test: number };
};

export default function Page() {
  const [data, setData] = useState<DataRow[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState<string>('label');
  const [trainPercent, setTrainPercent] = useState<number>(80);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ApiResponse | null>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const arr = evt.target?.result as ArrayBuffer;
      const wb = XLSX.read(arr, { type: 'array' });
      const wsname = wb.SheetNames[0];
      const ws = wb.Sheets[wsname];
      const json = XLSX.utils.sheet_to_json(ws, { defval: '' }) as DataRow[];
      setData(json);
      setResult(null);
      if (json.length > 0) {
        const cols = Object.keys(json[0]);
        setColumns(cols);
        if (cols.includes('label')) setTargetCol('label');
        else setTargetCol(cols[cols.length - 1]);
      } else {
        setColumns([]);
      }
    };
    reader.readAsArrayBuffer(f);
  }

  function remapTarget(raw: DataRow[], target: string): DataRow[] {
    if (target === 'label') return raw.map((r) => ({ ...r }));
    return raw.map((r) => {
      const copy: DataRow = { ...r };
      // create label key from selected target column
      copy.label = String(copy[target]);
      return copy;
    });
  }

  async function handleTrain() {
    if (!data) return alert('Upload file Excel terlebih dahulu');
    setLoading(true);
    try {
      const payload = remapTarget(data, targetCol);
      const res = await fetch('/api/naive-bayes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: payload, trainPercent })
      });
      const json = (await res.json()) as ApiResponse & { error?: string };
      if (!res.ok) throw new Error(json.error || 'Server error');
      setResult(json);
    } catch (err) {
      alert(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-8 bg-slate-50">
      <div className="max-w-4xl mx-auto bg-gray-50 p-6 rounded-2xl shadow-lg">
        <h1 className="text-2xl font-bold mb-4">UAS Data Mining Naive Bayes</h1>

        <label className="block mb-2">Upload file Excel (.xlsx). Sheet pertama akan digunakan</label>
        <input className="block mb-4 shadow-sm border border-gray-300 w-45" type="file" accept=".xlsx,.xls" onChange={handleFile} />

        {data && (
          <div className="mb-4">
            <div className="mb-2">Preview 5 baris pertama (total {data.length})</div>
            <div className="overflow-auto border rounded p-2">
              <table className="table-auto w-full text-sm border-collapse">
                <thead>
                  <tr>
                    {Object.keys(data[0]).map((k) => (
                      <th key={k} className="border px-2 py-1 text-left">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.slice(0, 5).map((row, i) => (
                    <tr key={i}>
                      {Object.keys(row).map((k) => (
                        <td key={k} className="border px-2 py-1">{String(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {columns.length > 0 && (
          <div className="mb-4">
            <label className="block mb-1">Pilih kolom target (label)</label>
            <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)} className="border rounded px-2 py-1">
              {columns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            <div className="text-xs text-slate-600 mt-1">Jika kolom target nama-nya bukan 'label', pilih di sini lalu sistem akan membuat field 'label' dari kolom terpilih.</div>
          </div>
        )}

        <div className="mb-4">
          <label className="block mb-1">Pilih persentase data untuk training: {trainPercent}%</label>
          <input
            type="range"
            min={10}
            max={90}
            step={10}
            value={trainPercent}
            onChange={(e) => setTrainPercent(Number(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-slate-600 mt-1">Data training: {trainPercent}%  data testing: {100 - trainPercent}%</div>
        </div>

        {/* BUTTON Train (Naive Bayes) */}
        <div className="flex gap-2">
          <button
            className="px-4 py-2 rounded bg-black text-white hover:bg-gray-600"
            onClick={handleTrain}
            disabled={loading || !data}
          >
            {loading ? 'Melatih...' : 'Train (Naive Bayes)'}
          </button>
        </div>

        {result && (
          <section className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Hasil Evaluasi</h2>
            <div className="mb-2">Total: {result.counts.total}  Train: {result.counts.train}  Test: {result.counts.test}</div>
            <div className="mb-2">Accuracy: {(result.eval.accuracy * 100).toFixed(2)}%</div>

            <div className="mb-4">
              <h3 className="font-medium">Confusion Matrix</h3>
              <div className="overflow-auto border rounded p-2 mt-2">
                <table className="table-auto text-sm">
                  <thead>
                    <tr>
                      <th className="border px-2 py-1">Actual \ Pred</th>
                      {result.model.classes.map((c) => (
                        <th key={c} className="border px-2 py-1">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.eval.confusionMatrix.map((row, i) => (
                      <tr key={i}>
                        <td className="border px-2 py-1 font-medium">{result.model.classes[i]}</td>
                        {row.map((v, j) => (
                          <td key={j} className="border px-2 py-1 text-center">{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="font-medium">Precision / Recall / F1 per kelas</h3>
              <table className="table-auto mt-2 text-sm">
                <thead>
                  <tr>
                    <th className="border px-2 py-1">Kelas</th>
                    <th className="border px-2 py-1">Precision</th>
                    <th className="border px-2 py-1">Recall</th>
                    <th className="border px-2 py-1">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {result.model.classes.map((c) => (
                    <tr key={c}>
                      <td className="border px-2 py-1">{c}</td>
                      <td className="border px-2 py-1">{(result.eval.precisionPerClass[c] * 100).toFixed(2)}%</td>
                      <td className="border px-2 py-1">{(result.eval.recallPerClass[c] * 100).toFixed(2)}%</td>
                      <td className="border px-2 py-1">{(result.eval.f1PerClass[c] * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="mt-2 text-sm text-slate-600">Macro Precision: {(result.eval.macroPrecision * 100).toFixed(2)}%  Macro Recall: {(result.eval.macroRecall * 100).toFixed(2)}%  Macro F1: {(result.eval.macroF1 * 100).toFixed(2)}%</div>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}