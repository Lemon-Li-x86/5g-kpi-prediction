"""Verify W4 saved evidence and unchanged six-page Web integration."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/final_audit_20261025'


def verify_w4():
    f=pd.read_csv(ROOT/'reports/w4_anomaly_test_flags.csv')
    keys=['Time','base_station','cell','beam']
    assert len(f)==483840 and not f.duplicated(keys).any()
    a=f.IsolationForest.astype(bool);b=f['3sigma_any_KPI'].astype(bool);c=f.IQR_any_KPI.astype(bool)
    candidate=a&(b|c)
    checks=dict(n=len(f),sigma=int(b.sum()),iqr=int(c.sum()),IF=int(a.sum()),consensus=int(candidate.sum()),
                IF_IQR=int((a&c).sum()),IF_sigma=int((a&b).sum()),three_OR=int((a|b|c).sum()),
                consensus_extra_vs_IF_IQR=int((candidate&~(a&c)).sum()))
    assert [checks[k] for k in ('sigma','iqr','IF','consensus','IF_IQR','IF_sigma','three_OR')]==[22059,51318,9900,3165,3159,2080,58108]
    assert np.array_equal(candidate, f.Consensus_IF_AND_sigma_OR_IQR.astype(bool))
    from src.core.config import KPIS
    corr=f.loc[candidate,KPIS].corr().to_numpy()
    tests=pd.read_csv(ROOT/'reports/w4_anomaly_kpi_correlation_tests.csv')
    assert len(tests)==6 and (tests.n_candidate_beam_hours==3165).all() and (tests.n_base_station_clusters==30).all()
    saved=pd.read_csv(ROOT/'reports/w4_anomaly_kpi_correlation_matrix.csv',index_col=0).loc[KPIS,KPIS].to_numpy()
    checks['max_correlation_difference']=float(np.abs(corr-saved).max())
    assert checks['max_correlation_difference']<1e-6
    checks['p_values_and_bootstrap']='reused existing W4 outputs; not re-estimated'
    (OUT/'w4_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('W4 verified',checks,flush=True)


def verify_web():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    pages=['数据管理与预处理','模型训练','模型管理与实验记录','KPI 预测','异常检测','KPI 关联分析']
    results=[]
    for page in pages:
        at.sidebar.radio[0].set_value(page).run()
        errors=[str(e.message) for e in at.exception]
        results.append(dict(page=page,exceptions=errors,visible_errors=[str(e.value) for e in at.error]))
        print('APPTEST',page,errors,flush=True)
    (OUT/'web_apptest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not any(r['exceptions'] for r in results)


if __name__=='__main__':
    verify_w4();verify_web()
