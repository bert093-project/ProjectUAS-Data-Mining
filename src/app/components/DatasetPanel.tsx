// src/app/components/DatasetPanel.tsx
import React from 'react';
import * as XLSX from 'xlsx';
import { detectColumnTypes, inferCandidateTarget } from '../utils/dataUtils';

type DataRow = Record<string, unknown>;

export default function DatasetPanel(props: {
  data: DataRow[] | null;
  onDataLoaded: (data: DataRow[]) => void;
  onColumnsDetected: (cols: string[]) => void;
  onColumnTypes: (types: Record<string, 'numeric'|'categorical'>) => void;
}) {
  const { data, onDataLoaded, onColumnsDetected, onColumnTypes } = props;

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
      onDataLoaded(json);
      const cols = json.length ? Object.keys(json[0]) : [];
      onColumnsDetected(cols);
      const types = detectColumnTypes(json);
      onColumnTypes(types);
    };
    reader.readAsArrayBuffer(f);
  }

  return (
    <div className="p-4">
      <h3 className="font-semibold mb-2">Dataset</h3>
      <input type="file" accept=".xlsx,.xls" onChange={handleFile} className="mb-5 px-10 py-10 border border-black bg-gray-50 text-black text-center rounded shadow-sm hover:bg-gray-100" />
      {data && (
        <div className="overflow-auto border rounded p-2">
          <table className="table-auto w-full text-sm">
            <thead>
              <tr>
                {Object.keys(data[0]).map(c => <th key={c} className="border px-2 py-1 text-left">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.slice(0,10).map((r, i) => (
                <tr key={i}>
                  {Object.keys(r).map(k => <td key={k} className="border px-2 py-1">{String((r as Record<string, unknown>)[k] ?? '')}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-xs text-slate-600 mt-2">Preview {Math.min(10, data.length)} rows (total {data.length})</div>
        </div>
      )}
    </div>
  );
}