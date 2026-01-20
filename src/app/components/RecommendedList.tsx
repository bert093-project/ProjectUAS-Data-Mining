// src/app/components/RecommendedList.tsx
import React from 'react';

type DataRow = Record<string, unknown>;
type ScoredItem = DataRow & { predictedLabel: string; probabilities: Record<string, number> };

export default function RecommendedList({ items }: { items: ScoredItem[] }) {
  if (!items || items.length === 0) return <div className="text-sm text-slate-600">No recommendations yet.</div>;
  return (
    <div className="space-y-3">
      {items.map((it, idx) => (
        <div key={idx} className="border rounded p-3 bg-white shadow-sm">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="font-semibold text-lg">{String((it as Record<string, unknown>)['brand'] ?? '')} {String((it as Record<string, unknown>)['model'] ?? '')}</div>
              <div className="text-sm text-slate-600">Price: Rp {Number((it as Record<string, unknown>)['price_idr'] ?? 0).toLocaleString()}</div>
              <div className="text-sm text-slate-600">Predicted class: {it.predictedLabel}</div>
              <div className="text-sm text-slate-600">Score: {((it.probabilities?.[it.predictedLabel] ?? 0) * 100).toFixed(2)}%</div>
            </div>

            <div className="w-1/3 text-xs">
              <div className="font-medium mb-1">Full specs</div>
              <ul className="space-y-1">
                {Object.entries(it).map(([k,v]) => {
                  if (k === 'predictedLabel' || k === 'probabilities') return null;
                  return (
                    <li key={k} className="flex justify-between">
                      <span className="font-medium">{k}</span>
                      <span className="text-right">{String(v ?? '')}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}