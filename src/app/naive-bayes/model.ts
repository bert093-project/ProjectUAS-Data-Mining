// src/app/naive-bayes/model.ts
// Naive Bayes (hybrid) that accepts featureColumns & targetColumn
export type DataRow = Record<string, unknown>;

type NumericStats = { mean: number; variance: number; count: number; };
type FeatureStats = { isNumeric: boolean; numeric?: Record<string, NumericStats>; categorical?: Record<string, Record<string, number>>; };

export type Model = { classPriors: Record<string, number>; features: Record<string, FeatureStats>; classes: string[]; featureColumns: string[]; targetColumn: string; };

function isNumericValue(v: unknown): boolean {
  if (v === null || v === undefined) return false;
  if (typeof v === 'number') return Number.isFinite(v);
  const s = String(v).replace(/[^\d.-]/g, '');
  const n = parseFloat(s);
  return Number.isFinite(n);
}

function parseNumber(v: unknown): number {
  if (v === null || v === undefined) return NaN;
  if (typeof v === 'number') return v;
  const s = String(v).replace(/[^\d.-]/g, '');
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : NaN;
}

function mean(nums: number[]): number { return nums.reduce((a,b)=>a+b,0) / nums.length; }
function variance(nums: number[], mu: number): number {
  const s = nums.reduce((a,b)=> a + (b-mu)*(b-mu), 0);
  return s / nums.length || 1e-9;
}

export function trainNaiveBayes(data: DataRow[], featureColumns: string[], targetColumn: string): Model {
  if (!Array.isArray(data) || data.length === 0) throw new Error('Empty data');
  // collect classes from targetColumn
  const classes = Array.from(new Set(data.map(r => String(r[targetColumn] ?? '')).filter(s => s !== '')));
  if (classes.length === 0) throw new Error('No classes found for targetColumn');

  const classCounts: Record<string, number> = {};
  for (const c of classes) classCounts[c] = 0;
  for (const r of data) {
    const cls = String(r[targetColumn] ?? '');
    if (cls === '') continue;
    classCounts[cls] = (classCounts[cls] || 0) + 1;
  }
  const total = data.length;
  const classPriors: Record<string, number> = {};
  for (const c of classes) classPriors[c] = (classCounts[c] || 0) / total;

  const features: Record<string, FeatureStats> = {};
  for (const fname of featureColumns) {
    const vals = data.map(r => r[fname]);
    const numeric = vals.every(v => isNumericValue(v));
    if (numeric) {
      const numericStats: Record<string, NumericStats> = {};
      for (const cls of classes) {
        const nums = data.filter(r => String(r[targetColumn] ?? '') === cls).map(r => parseNumber(r[fname])).filter(n => Number.isFinite(n));
        if (nums.length === 0) numericStats[cls] = { mean: 0, variance: 1, count: 0 };
        else numericStats[cls] = { mean: mean(nums), variance: variance(nums, mean(nums)), count: nums.length };
      }
      features[fname] = { isNumeric: true, numeric: numericStats };
    } else {
      const catStats: Record<string, Record<string, number>> = {};
      for (const cls of classes) catStats[cls] = {};
      for (const r of data) {
        const cls = String(r[targetColumn] ?? '');
        if (cls === '') continue;
        const v = String(r[fname] ?? '');
        catStats[cls][v] = (catStats[cls][v] || 0) + 1;
      }
      features[fname] = { isNumeric: false, categorical: catStats };
    }
  }

  return { classPriors, features, classes, featureColumns, targetColumn };
}

function gaussianProb(x: number, mu: number, varr: number): number {
  const denom = Math.sqrt(2 * Math.PI * varr) || 1e-9;
  const num = Math.exp(-((x - mu)*(x - mu)) / (2 * varr));
  return num / denom;
}

export function predict(model: Model, row: DataRow): { label: string; scores: Record<string, number> } {
  const scores: Record<string, number> = {};
  for (const cls of model.classes) {
    let logProb = Math.log(model.classPriors[cls] ?? 1e-9);
    for (const fname of model.featureColumns) {
      const f = model.features[fname];
      const val = row[fname];
      if (!f) continue;
      if (f.isNumeric && f.numeric) {
        const x = parseNumber(val);
        const stats = f.numeric[cls] ?? { mean: 0, variance: 1, count: 0 };
        if (!Number.isFinite(x)) logProb += Math.log(1e-6);
        else {
          const p = gaussianProb(x, stats.mean, stats.variance) || 1e-9;
          logProb += Math.log(p);
        }
      } else if (!f.isNumeric && f.categorical) {
        const counts = f.categorical[cls] || {};
        const v = String(val ?? '');
        const count = counts[v] || 0;
        const total = Object.values(counts).reduce((a,b)=>a+b,0);
        const V = Math.max(1, Object.keys(counts).length);
        const p = (count + 1) / (total + V);
        logProb += Math.log(p);
      }
    }
    scores[cls] = logProb;
  }
  const entries = Object.entries(scores);
  if (entries.length === 0) return { label: '', scores };
  const predicted = entries.sort((a,b) => b[1] - a[1])[0][0];
  return { label: predicted, scores };
}