// src/app/api/naive-bayes/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { DataRow, trainNaiveBayes, predict, Model } from '../../naive-bayes/model';

function shuffle<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function split<T>(arr: T[], trainPercent: number): { train: T[]; test: T[] } {
  const shuffled = shuffle(arr);
  const trainCount = Math.max(1, Math.floor((trainPercent / 100) * arr.length));
  return { train: shuffled.slice(0, trainCount), test: shuffled.slice(trainCount) };
}

function evaluate(model: Model, testData: DataRow[]) {
  const labels = model.classes;
  const labelIndex: Record<string, number> = {};
  labels.forEach((l, i) => (labelIndex[l] = i));
  const n = labels.length;
  const cm = Array.from({ length: n }, () => Array(n).fill(0));
  let correct = 0;
  for (const row of testData) {
    const actual = String(row.label);
    const pred = predict(model, row).label;
    const ai = labelIndex[actual];
    const pi = labelIndex[pred];
    cm[ai][pi]++;
    if (actual === pred) correct++;
  }
  const accuracy = testData.length ? correct / testData.length : 0;

  const precisionPerClass: Record<string, number> = {};
  const recallPerClass: Record<string, number> = {};
  const f1PerClass: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    const tp = cm[i][i];
    const fp = cm.reduce((s, row) => s + row[i], 0) - tp;
    const fn = cm[i].reduce((s, v) => s + v, 0) - tp;
    const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
    const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    precisionPerClass[labels[i]] = prec;
    recallPerClass[labels[i]] = rec;
    f1PerClass[labels[i]] = f1;
  }
  const macroPrecision = Object.values(precisionPerClass).reduce((a, b) => a + b, 0) / n;
  const macroRecall = Object.values(recallPerClass).reduce((a, b) => a + b, 0) / n;
  const macroF1 = Object.values(f1PerClass).reduce((a, b) => a + b, 0) / n;

  return { accuracy, confusionMatrix: cm, precisionPerClass, recallPerClass, f1PerClass, macroPrecision, macroRecall, macroF1 };
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const raw: DataRow[] = body.data;
    const trainPercent: number = Number(body.trainPercent) || 80;
    if (!Array.isArray(raw) || raw.length === 0) return NextResponse.json({ error: 'Data harus array non-kosong' }, { status: 400 });

    const { train, test } = split<DataRow>(raw, trainPercent);
    const model = trainNaiveBayes(train);
    const evalRes = evaluate(model, test);

    return NextResponse.json({ model, eval: evalRes, counts: { total: raw.length, train: train.length, test: test.length } });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}