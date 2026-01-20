// src/app/components/PerformancePanel.tsx
import React, { useMemo, useState } from 'react';

export default function PerformancePanel(props: {
  dataCount: number;
  featureColumns: string[];
  targetColumn: string | null;
  onRunEval: (cfg: { mode: 'split'; trainPercent: number }) => void;
}) {
  const { dataCount, featureColumns, targetColumn, onRunEval } = props;
  const [trainPercent, setTrainPercent] = useState<number>(80);

  // hitung persentase dan estimasi jumlah test rows (derived state)
  const { testPercent, trainRows, testRows } = useMemo(() => {
    const tp = Math.max(0, Math.min(100, 100 - trainPercent));
    const tr = Math.floor((trainPercent / 100) * dataCount);
    const te = Math.max(0, dataCount - tr);
    return { testPercent: tp, trainRows: tr, testRows: te };
  }, [trainPercent, dataCount]);

  function handleClick() {
    // DEBUG: always log when button clicked
    console.log('[PerformancePanel] Run button clicked');
    console.log('[PerformancePanel] state ->', {
      featureColumns,
      targetColumn,
      trainPercent,
      dataCount,
      testPercent,
      trainRows,
      testRows
    });

    // validation (show alert but still logged)
    if (!targetColumn) {
      console.warn('[PerformancePanel] Validation failed: no targetColumn');
      alert('Select target column in Process panel first');
      return;
    }
    if (!featureColumns || featureColumns.length === 0) {
      console.warn('[PerformancePanel] Validation failed: no featureColumns');
      alert('Select at least one feature column in Process panel');
      return;
    }

    try {
      onRunEval({ mode: 'split', trainPercent });
      console.log('[PerformancePanel] onRunEval invoked');
    } catch (err) {
      console.error('[PerformancePanel] onRunEval threw error:', err);
      alert('Failed to run evaluation: ' + (err instanceof Error ? err.message : String(err)));
    }
  }

  return (
    <div className="p-4">
      <h3 className="font-semibold mb-2">Performance / Evaluation (Train/Test split)</h3>

      <div className="mb-3">
        <label className="block text-sm mb-1">Train percentage: {trainPercent}%</label>
        <input
          type="range"
          min={10}
          max={90}
          step={5}
          value={trainPercent}
          onChange={(e) => setTrainPercent(Number(e.target.value))}
          className="w-full"
        />
        <div className="mt-2 text-md text-slate-600">
          <div>Data total: {dataCount} rows</div>
          <div>Training: {trainPercent}% (~{trainRows} rows)</div>
          <div>Testing: {testPercent}% (~{testRows} rows)</div>
        </div>
      </div>

      <div className="mt-3">
        {/* tombol memakai debug handler; validasi di dalam handleClick */}
        <button
          className="px-4 py-2 rounded bg-black text-white transition duration-100 hover:bg-gray-600"
          onClick={handleClick}
        >
          Run Evaluation
        </button>
      </div>
    </div>
  );
}