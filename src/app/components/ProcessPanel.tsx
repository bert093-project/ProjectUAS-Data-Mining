import React from 'react';

type Props = {
  columns: string[];
  featureColumns: string[];
  targetColumn: string | null;
  setFeatureColumns: (cols: string[]) => void;
  setTargetColumn: (col: string | null) => void;
  columnTypes: Record<string, 'numeric'|'categorical'>;
};

export default function ProcessPanel({ columns, featureColumns, targetColumn, setFeatureColumns, setTargetColumn, columnTypes }: Props) {
  function toggleAsTarget(col: string) {
    if (targetColumn === col) setTargetColumn(null);
    else setTargetColumn(col);
    // if target set, remove it from featureColumns
    setFeatureColumns(featureColumns.filter(c => c !== col));
  }
  function toggleFeature(col: string) {
    if (featureColumns.includes(col)) setFeatureColumns(featureColumns.filter(c => c !== col));
    else {
      // prevent adding target as feature
      if (targetColumn === col) return;
      setFeatureColumns([...featureColumns, col]);
    }
  }

  return (
    <div className="p-4">
      <h3 className="font-semibold mb-2">Process (features & target)</h3>

      <div className="mb-2 text-sm">
        <span className="inline-block px-2 py-1 bg-blue-50 text-blue-800 rounded mr-2">Feature</span>
        <span className="inline-block px-2 py-1 bg-yellow-100 text-yellow-800 rounded mr-2">Target</span>
        <span className="inline-block px-2 py-1 bg-white text-slate-600 rounded border">Unassigned</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {columns.map(col => {
          const isTarget = targetColumn === col;
          const isFeature = featureColumns.includes(col);
          const type = columnTypes[col] ?? 'categorical';
          return (
            <div key={col} className={`p-2 border rounded flex justify-between items-center ${isTarget ? 'bg-yellow-50 border-yellow-200' : isFeature ? 'bg-blue-50 border-blue-100' : ''}`}>
              <div>
                <div className="font-medium">{col}</div>
                <div className="text-xs text-slate-600">type: {type}</div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => toggleFeature(col)} className="px-2 py-1 text-xs bg-white border rounded">Feature</button>
                <button onClick={() => toggleAsTarget(col)} className="px-2 py-1 text-xs bg-white border rounded">Target</button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}