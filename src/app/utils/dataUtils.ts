// src/utils/dataUtils.ts
import type { DataRow } from '../naive-bayes/model';

// deteksi tipe kolom: 'numeric' atau 'categorical'
export function detectColumnTypes(data: DataRow[], sampleSize = 50): Record<string, 'numeric'|'categorical'> {
  const types: Record<string, 'numeric'|'categorical'> = {};
  if (!data || data.length === 0) return types;
  const sample = data.slice(0, Math.min(sampleSize, data.length));
  const cols = Object.keys(sample[0]);
  for (const c of cols) {
    let numericCount = 0;
    for (const r of sample) {
      const v = r[c];
      if (v === null || v === undefined) continue;
      const s = String(v).replace(/[^\d.-]/g,'');
      if (s === '') continue;
      const n = parseFloat(s);
      if (!Number.isNaN(n) && Number.isFinite(n)) numericCount++;
    }
    types[c] = numericCount >= Math.max(1, Math.floor(sample.length * 0.7)) ? 'numeric' : 'categorical';
  }
  return types;
}

// heuristik pilih kandidat target: column with small number of unique values or named 'price'
export function inferCandidateTarget(data: DataRow[]): string | null {
  if (!data || data.length === 0) return null;
  const cols = Object.keys(data[0]);
  // prefer exact matches
  const prefer = ['label','price_idr','price','class','price'];
  for (const p of prefer) if (cols.includes(p)) return p;
  // else pick column with low unique ratio
  let best: string | null = null;
  let bestRatio = 1;
  for (const c of cols) {
    const uniq = new Set(data.map(r => String(r[c] ?? '')).filter(s => s !== ''));
    const ratio = uniq.size / data.length;
    if (ratio < bestRatio) { bestRatio = ratio; best = c; }
  }
  return best;
}

// k-fold indices
export function kFoldIndices(n: number, k: number): number[][] {
  if (k <= 1) return [Array.from({length:n}, (_,i) => i)];
  const idx = Array.from({length:n}, (_,i) => i);
  // simple shuffle
  for (let i = idx.length -1; i>0; i--) { const j = Math.floor(Math.random()*(i+1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const folds: number[][] = Array.from({length:k}, ()=>[]);
  for (let i=0;i<n;i++) folds[i % k].push(idx[i]);
  return folds;
}

// split by trainPercent
export function splitData<T>(arr: T[], trainPercent: number): { train: T[]; test: T[] } {
  const a = arr.slice();
  for (let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
  const t = Math.max(1, Math.floor((trainPercent/100)*a.length));
  return { train: a.slice(0,t), test: a.slice(t) };
}

// metrics utilities: confusion matrix & precision/recall/f1
export function computeMetrics(classes: string[], actuals: string[], preds: string[]) {
  const idx: Record<string, number> = {};
  classes.forEach((c,i)=> idx[c]=i);
  const n = classes.length;
  const cm = Array.from({length:n}, ()=>Array(n).fill(0));
  for (let i=0;i<actuals.length;i++){
    const a = actuals[i]; const p = preds[i];
    const ai = idx[a]; const pi = idx[p];
    if (ai===undefined || pi===undefined) continue;
    cm[ai][pi] ++;
  }
  let correct = 0;
  for (let i=0;i<actuals.length;i++) if (actuals[i] === preds[i]) correct++;
  const accuracy = actuals.length ? correct / actuals.length : 0;

  const precisionPerClass: Record<string, number> = {};
  const recallPerClass: Record<string, number> = {};
  const f1PerClass: Record<string, number> = {};
  for (let i=0;i<n;i++){
    const tp = cm[i][i];
    const fp = cm.reduce((s,row)=> s + row[i],0) - tp;
    const fn = cm[i].reduce((s,v)=> s+v,0) - tp;
    const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
    const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    precisionPerClass[classes[i]] = prec;
    recallPerClass[classes[i]] = rec;
    f1PerClass[classes[i]] = f1;
  }
  const macroPrecision = Object.values(precisionPerClass).reduce((a,b)=>a+b,0) / (n||1);
  const macroRecall = Object.values(recallPerClass).reduce((a,b)=>a+b,0) / (n||1);
  const macroF1 = Object.values(f1PerClass).reduce((a,b)=>a+b,0) / (n||1);

  return { accuracy, confusionMatrix: cm, precisionPerClass, recallPerClass, f1PerClass, macroPrecision, macroRecall, macroF1 };
}