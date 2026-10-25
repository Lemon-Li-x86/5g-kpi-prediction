"""Round-trip actual new checkpoints through unchanged public inference API."""
from pathlib import Path
import sys
import json
from io import StringIO
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.core.config import KPIS,KEYS
from src.models.inference import predict_kpi,detect_anomalies

directory=ROOT/'resources/final_audit_20261025/holdout_seed42'
saved=np.load(directory/'paired_predictions.npz')
raw=pd.read_csv(ROOT/'data/processed/kpi_timeseries.csv',usecols=['Time',*KEYS,*KPIS])
key=pd.read_csv(directory/'beam_keys.csv').iloc[0]
mask=np.ones(len(raw),dtype=bool)
for k in KEYS:mask&=raw[k].eq(key[k])
history=raw.loc[mask & raw.Time.between(503,671)]
index=np.flatnonzero((saved['beam_index']==0)&(saved['target_time']==672))[0]
checks=[]
for name,artifact in [('GRU',directory/'gru.pt')]+[(n,directory/f'{n}_{k}.pkl') for n in ['RandomForest','XGBoost'] for k in KPIS]:
    predicted=predict_kpi(history,artifact)
    columns=[k for k in KPIS if k in predicted]
    expected=saved[name][index,[KPIS.index(k) for k in columns]]
    np.testing.assert_allclose(predicted[columns].to_numpy()[0],expected,rtol=1e-5,atol=2e-5)
    assert int(predicted.Time.iloc[0])==672
    decoded=pd.read_csv(StringIO(predicted.to_csv(index=False)))
    assert len(decoded)==1
    checks.append(dict(model=name,artifact=str(artifact.relative_to(ROOT)),roundtrip=True,csv_export=True))
anomalies=list((ROOT/'resources/models').rglob('anomalies.pkl'))
if not anomalies:raise RuntimeError('No historical published anomaly bundle')
values=raw[raw.Time.between(672,674)]
detected=detect_anomalies(values,anomalies[0])
assert len(detected)==5760 and not detected.duplicated([*KEYS,'Time']).any()
assert (detected.consensus==(detected.isolation_forest&(detected.sigma|detected.iqr))).all()
checks.append(dict(model='historical anomaly',artifact=str(anomalies[0].relative_to(ROOT)),n=len(detected),
                   csv_export=len(pd.read_csv(StringIO(detected.to_csv(index=False))))==5760))
(ROOT/'reports/final_audit_20261025/real_artifact_roundtrip.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Passed actual artifact inference/CSV:',len(checks))
