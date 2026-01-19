// src/app/naive-bayes/model.ts
export type DataRow = Record<string, string | number>;

type NumericStats = {
  mean: number;
  variance: number;
  count: number;
};

type FeatureStats = {
  isNumeric: boolean;
  numeric?: Record<string, NumericStats>;
  categorical?: Record<string, Record<string, number>>;
};

export type Model = {
  classPriors: Record<string, number>;
  features: Record<string, FeatureStats>;
  classes: string[];
};

function isNumericColumn(values: (string | number)[]): boolean {
  for (const v of values) {
    if (v === null || v === undefined) return false;
    const n = typeof v === 'number' ? v : parseFloat(String(v));
    if (!Number.isFinite(n)) return false;
  }
  return true;
}

function mean(nums: number[]): number {
  const s = nums.reduce((a, b) => a + b, 0);
  return s / nums.length;
}

function variance(nums: number[], mu: number): number {
  const s = nums.reduce((a, b) => a + (b - mu) * (b - mu), 0);
  return s / nums.length || 1e-9;
}

export function trainNaiveBayes(data: DataRow[]): Model {
  if (data.length === 0) throw new Error('Data kosong');
  if (!('label' in data[0])) throw new Error('Kolom target harus bernama "label"');

  const featureNames = Object.keys(data[0]).filter((k) => k !== 'label');
  const classes = Array.from(new Set(data.map((r) => String(r.label))));

  const classCounts: Record<string, number> = {};
  for (const c of classes) classCounts[c] = 0;
  for (const row of data) classCounts[String(row.label)]++;

  const classPriors: Record<string, number> = {};
  for (const c of classes) classPriors[c] = classCounts[c] / data.length;

  const features: Record<string, FeatureStats> = {};
  for (const fname of featureNames) {
    const colValues = data.map((r) => r[fname]);
    const numeric = isNumericColumn(colValues);
    if (numeric) {
      const numericStats: Record<string, NumericStats> = {};
      for (const cls of classes) {
        const nums = data.filter((r) => String(r.label) === cls).map((r) => Number(r[fname]));
        const mu = mean(nums);
        const v = variance(nums, mu);
        numericStats[cls] = { mean: mu, variance: v, count: nums.length };
      }
      features[fname] = { isNumeric: true, numeric: numericStats };
    } else {
      const catStats: Record<string, Record<string, number>> = {};
      for (const cls of classes) catStats[cls] = {};
      for (const row of data) {
        const cls = String(row.label);
        const val = String(row[fname]);
        catStats[cls][val] = (catStats[cls][val] || 0) + 1;
      }
      features[fname] = { isNumeric: false, categorical: catStats };
    }
  }

  return { classPriors, features, classes };
}

function gaussianProb(x: number, mu: number, varr: number): number {
  const denom = Math.sqrt(2 * Math.PI * varr) || 1e-9;
  const num = Math.exp(-((x - mu) * (x - mu)) / (2 * varr));
  return num / denom;
}

export function predict(model: Model, row: DataRow): { label: string; scores: Record<string, number> } {
  const scores: Record<string, number> = {};
  for (const cls of model.classes) {
    let logProb = Math.log(model.classPriors[cls] || 1e-9);
    for (const fname of Object.keys(model.features)) {
      const f = model.features[fname];
      const val = row[fname];
      if (f.isNumeric && f.numeric) {
        const x = Number(val);
        const stats = f.numeric[cls];
        const p = gaussianProb(x, stats.mean, stats.variance) || 1e-9;
        logProb += Math.log(p);
      } else if (!f.isNumeric && f.categorical) {
        const counts = f.categorical[cls] || {};
        const v = String(val);
        const count = counts[v] || 0;
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        const V = Math.max(1, Object.keys(counts).length);
        const p = (count + 1) / (total + V);
        logProb += Math.log(p);
      }
    }
    scores[cls] = logProb;
  }
  const predicted = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  return { label: predicted, scores };
}