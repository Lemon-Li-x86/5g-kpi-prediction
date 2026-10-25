"""Fixed teacher-review protocol, isolated from Web and historical scores.

Run: python -m src.analysis.final_model_validation --seeds 42 43
Uses matched 60,000 training labels for RF/XGB/GRU, temporal inner validation,
three outer folds and a final holdout. Does not tune to test p-values.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.core.config import ROOT, KPIS, KEYS
from src.models.train_deep_learning_models import (
    load_tensor, fit_training_prefix, WindowDataset, sample_ids, evaluate_loss, forecast,
)
from src.models.architectures import ForecastNet
from src.models.train_traditional_models import build_models
from src.analysis.validate_model_robustness import hac_paired_test, holm_adjust

OUT = ROOT / 'reports/final_audit_20261025/experiments'
ARTIFACTS = ROOT / 'resources/final_audit_20261025'


def design_at(z, beams, targets):
    """Exact KPI-grouped 20-column contract; targets are array positions T."""
    return np.column_stack([z[beams, targets - 1 - lag, j]
                            for j in range(len(KPIS)) for lag in (0, 1, 2, 24, 168)])


def loss_by_hour(y, pred, hours, model, seed, scenario):
    delta = pred.astype(np.float64) - y.astype(np.float64)
    rows = []
    for hour in np.unique(hours):
        mask = hours == hour
        for j, kpi in enumerate(KPIS):
            rows.append(dict(scenario=scenario, seed=seed, model=model, KPI=kpi,
                             Time=int(hour), n=int(mask.sum()),
                             mae_loss=float(np.abs(delta[mask, j]).mean()),
                             mse_loss=float(np.square(delta[mask, j]).mean())))
    return rows


def paired_table(losses):
    """One prespecified Holm family of 24 tests per scenario/seed."""
    rows = []
    for (scenario, seed), group in losses.groupby(['scenario', 'seed']):
        family = []
        for reference, contender in [('RandomForest', 'GRU'), ('XGBoost', 'GRU'), ('RandomForest', 'XGBoost')]:
            for kpi in KPIS:
                for loss in ('mae_loss', 'mse_loss'):
                    subset = group[group.KPI == kpi]
                    wide = subset.pivot(index='Time', columns='model', values=loss).sort_index()
                    counts = subset.pivot(index='Time', columns='model', values='n')
                    if wide[[reference, contender]].isna().any().any() or not (counts[reference] == counts[contender]).all():
                        raise ValueError('Unpaired hours or beam denominators')
                    test = hac_paired_test(wide[reference], wide[contender], bandwidth=24)
                    family.append(dict(scenario=scenario, seed=int(seed), reference=reference,
                                       contender=contender, KPI=kpi, loss=loss, bandwidth=24, **test))
        frame = pd.DataFrame(family)
        frame['holm_p'] = holm_adjust(frame.p_two_sided)
        frame['holm_family_size'] = 24
        rows.extend(frame.to_dict('records'))
    return pd.DataFrame(rows)


def main():
    global OUT, ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--max-train', type=int, default=60000)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT,
                        help='New empty report directory; existing experiment is never replaced')
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS,
                        help='New empty local model/prediction directory')
    args = parser.parse_args()
    OUT, ARTIFACTS = args.out, args.artifacts
    OUT.mkdir(parents=True, exist_ok=False)
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    csv = ROOT / 'data/processed/kpi_timeseries.csv'
    _, raw, timeline = load_tensor(csv)
    keys = pd.read_csv(csv, usecols=KEYS).drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    protocol = dict(version=1, seeds=args.seeds, sample_seed=42, max_train=args.max_train,
                    epochs=args.epochs, patience=3, batch_size=256, learning_rate=.001,
                    hidden=32, seq_len=24, seasonal_lag=168, cpu_threads=args.threads,
                    pairing='same beam/target rows, all beams in each evaluation hour',
                    sample_budget='same training labels and time range for RF, XGB, GRU; different input representations',
                    folds=[[384,480,336],[480,576,432],[576,672,528]],
                    holdout=[672,840,588],
                    selection='fixed before running; no test-based parameter adjustment',
                    tests='24 comparisons per scenario/seed, two-sided NW HAC24, Holm all24',
                    uncertainty='CI is pointwise HAC95%; Holm applies to p-values, seeds are not independent new time periods',
                    limitation='outer test is not an official external test dataset; labels observed each hour for next-hour forecast')
    protocol['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT/'src/models/train_deep_learning_models.py',
                                           ROOT/'src/models/train_traditional_models.py', ROOT/'src/models/architectures.py']}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    env = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
               numpy=np.__version__, pandas=pd.__version__, device='cpu')
    (OUT/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    all_metrics, all_losses, all_runs = [], [], []
    scenarios = [('fold1',384,480,336,args.seeds[0]), ('fold2',480,576,432,args.seeds[0]),
                 ('fold3',576,672,528,args.seeds[0])] + [('holdout',672,840,588,s) for s in args.seeds]
    for scenario, start, end, inner, seed in scenarios:
        tag = f'{scenario}_seed{seed}'
        print(f'BEGIN {tag}: fit targets 169..{inner-1}, inner {inner}..{start-1}, evaluate {start}..{end-1}', flush=True)
        run_dir = ARTIFACTS/tag; run_dir.mkdir()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        z, mean, scale = fit_training_prefix(raw, timeline, inner)
        train_t = np.flatnonzero((timeline >=169)&(timeline <inner))
        inner_t = np.flatnonzero((timeline >=inner)&(timeline <start))
        eval_t = np.flatnonzero((timeline >=start)&(timeline <end))
        rng = np.random.default_rng(42)  # same sampled labels across model seeds
        ids = sample_ids(len(raw),len(train_t),args.max_train,rng)
        val_ids = sample_ids(len(raw),len(inner_t),12000,rng)
        eval_ids = np.arange(len(raw)*len(eval_t),dtype=np.int64)
        datasets = [WindowDataset(z,t,24,168,ii) for t,ii in [(train_t,ids),(inner_t,val_ids),(eval_t,eval_ids)]]
        loaders = [DataLoader(ds,batch_size=256,shuffle=(i==0),num_workers=0) for i,ds in enumerate(datasets)]
        b_train = ids//len(train_t); t_train=train_t[ids%len(train_t)]
        b_eval = eval_ids//len(eval_t); t_eval=eval_t[eval_ids%len(eval_t)]
        X = design_at(z,b_train,t_train); Xtest = design_at(z,b_eval,t_eval)
        ytrain = raw[b_train,t_train]; truth = raw[b_eval,t_eval].astype(np.float64)
        predictions = {}
        params = build_models(False)
        model_params = {}
        for name in ('RandomForest','XGBoost'):
            t0=time.perf_counter(); pred=np.empty_like(truth)
            for j,k in enumerate(KPIS):
                estimator=copy.deepcopy(params[name]).set_params(random_state=seed,n_jobs=args.threads)
                estimator.fit(X,ytrain[:,j]); pred[:,j]=estimator.predict(Xtest)
                model_params[name]=estimator.get_params()
                import pickle
                columns=[f'{kpi}_z_{suffix}' for kpi in KPIS for suffix in ('t','lag1','lag2','lag24','lag168')]
                with (run_dir/f'{name}_{k}.pkl').open('wb') as f:
                    pickle.dump(dict(version=1,family='ml',model=estimator,target=k,columns=columns,
                                     mean=mean.tolist(),scale=scale.tolist()),f)
            predictions[name]=pred
            print(f'{tag} {name}: {time.perf_counter()-t0:.1f}s',flush=True)
        # Reset initialization independently of ML fitting and data loader construction.
        torch.manual_seed(seed)
        net=ForecastNet('gru',24,32); optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        best=float('inf'); patience=3; history=[]; checkpoint=run_dir/'gru.pt'
        t0=time.perf_counter()
        for epoch in range(args.epochs):
            net.train(); sums=0.; n=0
            for seq,seasonal,y,_,_ in loaders[0]:
                optimizer.zero_grad(set_to_none=True); p=net(seq,seasonal)
                loss=(p-y).square().mean(); loss.backward(); optimizer.step()
                sums+=float(loss.detach())*y.numel(); n+=y.numel()
            val=evaluate_loss(net,loaders[1],torch.device('cpu'))
            history.append(dict(epoch=epoch+1,train_mse=sums/n,validation_mse=val))
            print(f'{tag} GRU epoch{epoch+1}: train={sums/n:.6f} inner={val:.6f}',flush=True)
            if val < best-1e-6:
                best=val; best_epoch=epoch+1; patience=3; torch.save(net.state_dict(),checkpoint)
            else:
                patience-=1
                if patience==0: break
        net.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
        _,pred_z,_=forecast(net,loaders[2],torch.device('cpu'))
        predictions['GRU']=pred_z.astype(np.float64)*scale+mean
        checkpoint.with_suffix('.json').write_text(json.dumps(dict(version=1,family='dl',mode='gru',
                seq_len=24,seasonal_lag=168,hidden=32,mean=mean.tolist(),scale=scale.tolist(),
                scaler_fit_until=inner)),encoding='utf-8')
        run=dict(scenario=scenario,seed=seed,training=[169,inner-1],internal_validation=[inner,start-1],
                 evaluation=[start,end-1],train_samples=len(ids),validation_samples=len(val_ids),
                 evaluation_samples=len(eval_ids),scaler_fit_until_exclusive=inner,
                 mean=mean.tolist(),scale=scale.tolist(),best_epoch=best_epoch,loss_history=history,
                 model_parameters=model_params,GRU_seconds=time.perf_counter()-t0)
        all_runs.append(run)
        # Keep exact sample pairing and raw predictions for independent verification.
        np.savez_compressed(run_dir/'paired_predictions.npz',beam_index=b_eval,target_time=timeline[t_eval],
                            truth=truth,**predictions)
        keys.to_csv(run_dir/'beam_keys.csv',index=False)
        for name,pred in predictions.items():
            for j,k in enumerate(KPIS):
                all_metrics.append(dict(scenario=scenario,seed=seed,model=name,KPI=k,n=len(truth),
                        MAE=mean_absolute_error(truth[:,j],pred[:,j]),
                        RMSE=np.sqrt(mean_squared_error(truth[:,j],pred[:,j])),R2=r2_score(truth[:,j],pred[:,j])))
            all_losses.extend(loss_by_hour(truth,pred,timeline[t_eval],name,seed,scenario))
        pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
        losses=pd.DataFrame(all_losses);losses.to_csv(OUT/'hourly_losses.csv',index=False)
        (OUT/'runs.json').write_text(json.dumps(all_runs,indent=2),encoding='utf-8')
        if scenario=='holdout':
            paired_table(losses[losses.scenario=='holdout']).to_csv(OUT/'holdout_significance.csv',index=False)
        print(f'END {tag}',flush=True)
    metrics=pd.DataFrame(all_metrics)
    metrics[metrics.scenario.str.startswith('fold')].groupby(['model','KPI'])[['MAE','RMSE','R2']].agg(['mean','std']).to_csv(OUT/'walkforward_summary.csv')
    metrics[metrics.scenario=='holdout'].groupby(['model','KPI'])[['MAE','RMSE','R2']].agg(['mean','std']).to_csv(OUT/'seed_summary.csv')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(runs=len(all_runs),metrics=len(all_metrics))),encoding='utf-8')


if __name__=='__main__':
    main()
