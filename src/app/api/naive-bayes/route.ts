// src/app/api/naive-bayes/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { trainNaiveBayes, predict, DataRow } from '../../naive-bayes/model';
import { kFoldIndices, computeMetrics } from '@/app/utils/dataUtils';

// helper softmax
function softmax(logScores: number[]): number[] {
  const exps = logScores.map(s => Math.exp(s));
  const sum = exps.reduce((a,b)=>a+b, 0) || 1;
  return exps.map(e => e / sum);
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const raw: DataRow[] = Array.isArray(body.data) ? body.data : [];
    const featureColumns: string[] = Array.isArray(body.featureColumns) ? body.featureColumns : [];
    const targetColumn: string = String(body.targetColumn ?? '');
    const evalCfg = body.eval ?? { mode: 'split', trainPercent: 80 };

    if (raw.length === 0) return NextResponse.json({ error: 'Data array required' }, { status: 400 });
    if (!targetColumn) return NextResponse.json({ error: 'targetColumn required' }, { status: 400 });
    if (!Array.isArray(featureColumns) || featureColumns.length === 0) return NextResponse.json({ error: 'featureColumns required' }, { status: 400 });

    // ensure target values exist (filter rows that miss target)
    const cleaned = raw.filter(r => {
      const v = String(r[targetColumn] ?? '').trim();
      return v.length > 0;
    });
    if (cleaned.length === 0) return NextResponse.json({ error: 'No valid rows with target values' }, { status: 400 });

    const mode = String(evalCfg.mode ?? 'split');

    if (mode === 'split') {
      const trainPercent = Number(evalCfg.trainPercent) || 80;
      // shuffle + split
      const a = cleaned.slice();
      for (let i=a.length-1;i>0;i--){ const j = Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
      const tcount = Math.max(1, Math.floor((trainPercent/100) * a.length));
      const train = a.slice(0,tcount);
      const test = a.slice(tcount);

      const model = trainNaiveBayes(train, featureColumns, targetColumn);

      // evaluate
      const actuals: string[] = [];
      const preds: string[] = [];
      for (const row of test) {
        actuals.push(String(row[targetColumn] ?? ''));
        const pred = predict(model, row).label;
        preds.push(pred);
      }
      const evalRes = computeMetrics(model.classes, actuals, preds);

      // scored items for whole dataset (use model to score cleaned dataset)
      const scoredItems = cleaned.map(row => {
        const pr = predict(model, row);
        const logScores = model.classes.map(c => pr.scores[c] ?? -1e9);
        const probs = softmax(logScores);
        const probMap: Record<string, number> = {};
        model.classes.forEach((c,i)=> probMap[c] = probs[i]);
        return { ...row, predictedLabel: pr.label, probabilities: probMap };
      });

      return NextResponse.json({
        mode: 'split',
        counts: { total: cleaned.length, train: train.length, test: test.length },
        model: { classes: model.classes, featureColumns, targetColumn },
        eval: evalRes,
        scoredItems
      });
    } else if (mode === 'cv') {
      const k = Math.max(2, Math.min(10, Number(evalCfg.cvFolds) || 5));
      const n = cleaned.length;
      const folds = kFoldIndices(n, k); // array of index arrays
      const foldResults: any[] = [];

      for (let i=0;i<k;i++) {
        const testIdx = folds[i];
        const trainIdx = folds.flatMap((f, idx) => idx===i ? [] : f);
        const train = trainIdx.map(idx => cleaned[idx]);
        const test = testIdx.map(idx => cleaned[idx]);
        const model = trainNaiveBayes(train, featureColumns, targetColumn);
        const actuals: string[] = [];
        const preds: string[] = [];
        for (const row of test) {
          actuals.push(String(row[targetColumn] ?? ''));
          preds.push(predict(model, row).label);
        }
        const evalRes = computeMetrics(model.classes, actuals, preds);
        foldResults.push({ fold: i+1, counts: { train: train.length, test: test.length }, eval: evalRes });
      }

      // aggregate mean and std for accuracy
      const accuracies = foldResults.map(f => f.eval.accuracy);
      const meanAcc = accuracies.reduce((a,b)=>a+b,0) / accuracies.length;
      const stdAcc = Math.sqrt(accuracies.map(a=> (a-meanAcc)*(a-meanAcc)).reduce((s,x)=>s+x,0) / accuracies.length);

      // train final model on all data and score
      const finalModel = trainNaiveBayes(cleaned, featureColumns, targetColumn);
      const scoredItems = cleaned.map(row => {
        const pr = predict(finalModel, row);
        const logScores = finalModel.classes.map(c => pr.scores[c] ?? -1e9);
        const probs = softmax(logScores);
        const probMap: Record<string, number> = {};
        finalModel.classes.forEach((c,i)=> probMap[c] = probs[i]);
        return { ...row, predictedLabel: pr.label, probabilities: probMap };
      });

      return NextResponse.json({
        mode: 'cv',
        cvFolds: k,
        folds: foldResults,
        cvMean: { accuracy: meanAcc, accuracyStd: stdAcc },
        model: { classes: finalModel.classes, featureColumns, targetColumn },
        scoredItems
      });
    } else if (mode === 'manual') {
      const manual = evalCfg.manual ?? {};
      const trainIdx: number[] = Array.isArray(manual.trainIndices) ? manual.trainIndices : [];
      const testIdx: number[] = Array.isArray(manual.testIndices) ? manual.testIndices : [];
      if (trainIdx.length === 0 || testIdx.length === 0) return NextResponse.json({ error: 'manual mode requires trainIndices and testIndices' }, { status: 400 });
      const train = trainIdx.map((i:number)=> cleaned[i]).filter(Boolean);
      const test = testIdx.map((i:number)=> cleaned[i]).filter(Boolean);
      const model = trainNaiveBayes(train, featureColumns, targetColumn);
      const actuals: string[] = [];
      const preds: string[] = [];
      for (const row of test) {
        actuals.push(String(row[targetColumn] ?? ''));
        preds.push(predict(model, row).label);
      }
      const evalRes = computeMetrics(model.classes, actuals, preds);
      const scoredItems = cleaned.map(row => {
        const pr = predict(model, row);
        const logScores = model.classes.map(c => pr.scores[c] ?? -1e9);
        const probs = softmax(logScores);
        const probMap: Record<string, number> = {};
        model.classes.forEach((c,i)=> probMap[c] = probs[i]);
        return { ...row, predictedLabel: pr.label, probabilities: probMap };
      });
      return NextResponse.json({
        mode: 'manual',
        counts: { total: cleaned.length, train: train.length, test: test.length },
        model: { classes: model.classes, featureColumns, targetColumn },
        eval: evalRes,
        scoredItems
      });
    } else {
      return NextResponse.json({ error: 'Unknown eval mode' }, { status: 400 });
    }

  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}