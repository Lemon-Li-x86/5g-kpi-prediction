"""Create evidence-led closure documents from completed experiment outputs."""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'reports/final_audit_20261025'
EXP=AUDIT/'experiments'


def markdown(frame):
    # Avoid optional tabulate dependency.
    rows=[list(frame.columns),*frame.astype(str).values.tolist()]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])


def main():
    assert (EXP/'COMPLETE.json').is_file()
    metrics=pd.read_csv(EXP/'metrics.csv');tests=pd.read_csv(EXP/'holdout_significance.csv')
    assert len(metrics)==60 and len(tests)==48
    hold=metrics[(metrics.scenario=='holdout')&(metrics.seed==42)].copy()
    for col in ('MAE','RMSE','R2'):hold[col]=hold[col].map(lambda v:f'{v:.6f}')
    stats=tests[(tests.seed==42)&(tests.reference=='RandomForest')&(tests.contender=='GRU')].copy()
    stats['95% CI']=stats.apply(lambda r:f'[{r.ci95_low:.6f}, {r.ci95_high:.6f}]',axis=1)
    stats['mean_loss_difference']=stats.mean_loss_difference.map(lambda v:f'{v:.6f}')
    stats['holm_p']=stats.holm_p.map(lambda v:f'{v:.4g}')
    report='''# 最终模型实验与统计复核

日期：2026-10-25。源码基线0781ca9，当前本机CPU。固定协议执行5次场景，得到60组KPI指标、48组留出损失检验；没有按照p值调整参数。历史主表仍保留在原路径，不用本次抽样训练表覆盖旧成绩。

## 训练和评价协议

RF、XGBoost、GRU使用相同60,000条训练标签和训练目标时间；GRU为四输出，RF/XGB每KPI各拟合一次。固定树模型参数沿用主实验（RF30棵、XGB250轮），GRU单层32隐藏、24小时序列+季节向量、Adam .001、batch256、最多12轮、patience3。数据抽样seed42固定；模型seed42/43不同。输入表示、早停过程和计算开销仍不同，不称为等算力架构消融。

| 场景 | 训练目标 | 内部早停目标 | 外层评估目标 | 每模型外层记录 |
|---|---|---|---|---:|
| Fold1 | 169–335 | 336–383 | 384–479 | 276,480 |
| Fold2 | 169–431 | 432–479 | 480–575 | 276,480 |
| Fold3 | 169–527 | 528–575 | 576–671 | 276,480 |
| 留出 | 169–587 | 588–671 | 672–839 | 483,840 |

每折scaler只在内部早停起点前的raw观测拟合；训练集的标签及输入均早于内部验证与外层评估。外层评估不用来选择早停轮次。三折只运行seed42；留出运行42/43。每小时使用截至当时真实历史预测下一小时，不是一次性递归预测96/168小时。

## 留出指标 seed42

'''+markdown(hold[['KPI','model','MAE','RMSE','R2']])+'''

GRU四项MAE最低；DLPRB、DLThpTime、MR_number的RMSE/R²也领先。DLThpVol的RMSE/R²仍以RF较好。seed43维持上述方向，但两种seed只提供有限随机性复核，不是独立时间重复。

## 配对显著性

在同一beam及目标小时逐样本计算绝对/平方损失后，按小时平均，得168个配对值。d=候选损失−参考损失，负值有利于候选。Newey–West HAC Bartlett带宽24，双侧正态近似；每个scenario/seed预定24组（3对模型×4KPI×2损失）一起Holm校正。CI为逐项95%区间，不是Holm同时置信区间。

'''+markdown(stats[['KPI','loss','mean_loss_difference','95% CI','holm_p']])+'''

DLThpVol的GRU对RF两类损失在seed42均未显著（校正p≈0.354）；不能以MAE略低宣称全面胜出。其余三项GRU对RF的两种损失显著较低。GRU对XGB与RF对XGB、seed43全部结果见holdout_significance.csv；不挑选更好看的seed作为唯一结论。

## 决策规则与限制

以平均绝对误差优先，推荐GRU作为默认研究候选；对DLThpVol大误差敏感时保留RF。新三折均值的四项MAE/RMSE以GRU较低，但DLThpVol跨折RMSE方差较大。历史传统等权排名把RMSE和R²各计一次，会重复计入同一平方误差排序；本次不再以这两者的重复名次制造额外优势。

现有九种方法的历史完整比较表仍为reports/week3/w3_ml_dl_all_metrics.csv。本次仅复评GRU/RF/XGB，不能把另外六种方法的历史成绩与新预算成绩混排成同一次实验。

最终测试时段曾用于历史选型，因而本次为固定协议复核，不是新外部测试。统计检验基于168小时且HAC为近似；未处理一切可能的非平稳性。两种模型种子、三个相邻时间折均不支持无限推广。未重做LSTM/Transformer三折、全部模型自动调参、最佳Optuna参数全量复评、独立外部验证或真实故障标签评估。

## 复现与产物

本次运行：`.venv/Scripts/python.exe -u -m src.analysis.final_model_validation --seeds 42 43 --threads 4`。脚本拒绝复用已存在实验目录；重新复现使用`--out reports/final_audit_reproduction --artifacts resources/final_audit_reproduction`指定两个全新目录，不能删除旧结果后冒充同一次运行。本次执行时的原始runner另存experiments/runner_used.py，其SHA与protocol.json一致；当前runner只补充了输出目录CLI参数，不改变计算协议。

protocol.json记录固定参数、各折时间、检验族和源码哈希；runs.json记录实际epoch与scaler；metrics.csv、hourly_losses.csv、walkforward_summary.csv、seed_summary.csv及holdout_significance.csv为实际输出。逐beam-hour配对预测、beam键和模型在resources/final_audit_20261025/（Git忽略），本机备份不可省略。
'''
    (AUDIT/'模型实验与统计复核.md').write_text(report,encoding='utf-8')
    decisions=[]
    for model,g in metrics[(metrics.scenario=='holdout')&(metrics.seed==42)].groupby('model'):
        decisions.append(dict(model=model,training_labels=60000,decision='GRU MAE preferred; RF DLThpVol RMSE exception'))
    (EXP/'decision_context.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
    closure='''# 导师反馈逐项关闭清单

2026年10月25日最终审查。指定项目工作树为准。原Web入口、src/ui、src/services及core原文件受哈希保护，未修改；新算法只在已有接口下拒绝不合法小时输入，保留现有有效输入行为。

## 十项反馈的最终状态

| 原始要求 | 完成结果 | 可检查证据 | 最终状态及遗留限制 |
|---|---|---|---|
| ①20维输入说明 | 四KPI×t/lag1/2/24/168，t+1 raw标签；同beam对齐；训练推理列顺序一致 | tests/test_algorithm_audit.py；参数登记簿；修订W2及完整报告 | 已验证完成；原数据物理换算单位未提供，明确保留原尺度 |
| ②模型统一比较及综合选择 | 保留九方法历史对照，新RF/XGB/GRU在同60,000训练标签上复评；MAE优先GRU、DLThpVol RMSE保留RF | experiments/metrics.csv；模型实验与统计复核.md；历史week3/w3_ml_dl_all_metrics.csv | 已验证完成（固定协议范围）；训练时间/计算预算及输入表示不同，不宣称绝对最优 |
| ③三折时序验证与滚动回测 | GRU/RF/XGB三折、外层与内部早停分离；scaler只用内部训练前缀；缺小时加载器修复 | experiments/protocol.json、runs.json、walkforward_summary.csv；算法回归测试 | 已验证完成（重点三模型）；LSTM/Transformer及独立外部数据未新增 |
| ④统计显著性 | 同beam-hour损失按小时配对、HAC24、每seed24组Holm；保存效应/CI/p | experiments/hourly_losses.csv、holdout_significance.csv（48行） | 已验证完成；DLThpVol seed42 GRU对RF两类损失未显著，近似推断限制已披露 |
| ⑤超参数优化说明 | 逐模型固定设置、RF TPE12次范围/目标/256beam及最佳参数来源列明 | docs/参数登记簿_最终审查.md；historical tune JSON/CSV；修订报告 | 已验证完成（披露要求）；未要求也未虚构全九模型搜索，调优最优参数无新全量复评 |
| ⑥Table3、Figure2及全部图表 | 保留合格历史三指标表；完整报告跳号重排；历史实验图补样本/尺度图注；新增三指标、CV、效应区间表 | deliverables/review_20261025/十份修订Word及PDF；report_revision_manifest.json；版面QA | 已完成内容修订；最终版面验收见document_qa.json；旧原件保留归档不作本次交付 |
| ⑦异常标记率与分布解释 | 复核483840行及22059/51318/9900；复用逐beam偏度峰度、零IQR回退0，限定为支持性解释 | w4_verification.json；历史w4_anomaly_threshold_shape_summary.csv；修订异常报告 | 已验证完成；未唯一归因于偏态，未做新的漂移贡献分解 |
| ⑧AND/OR规则 | 实际标记复核IF∧(3σ∨IQR)=3165；IF∧IQR3159，IF∧3σ2080，全OR58108；增量6条 | w4_verification.json；原w4_anomaly_logic_comparison.csv；真实模型回载 | 已验证完成；无真值不称检测准确率提高 |
| ⑨相关矩阵n及显著性 | 3165记录/30站六对r、站级bootstrap95%CI、聚类稳健零斜率及Holm；矩阵重新核对一致 | 原w4_anomaly_kpi_correlation_tests.csv；w4_verification.json；修订根因报告 | 已验证完成（W4范围）；p/CI复用非本次重估，单向聚类和选择偏差仍有限制 |
| ⑩最终报告和Web集成一致 | 最新57项自动测试通过，六页AppTest无异常；9个新预测artifact+1个旧异常artifact接口/CSV检查，现有DL CLI烟测通过 | final_tests.txt、web_apptest.json、real_artifact_roundtrip.json、cli_smoke.txt、preservation_check.json | 自动回归已验证完成；未重做真人上传下载/完整WebSocket训练流程，不能称完整手工验收完成 |

## 实质修复及依据

1. DL和异常训练loader共同缺小时仍会按位置取lag：失败用例已复现，新增hourly_contract.validate_hourly_timeline，拒绝重复、断档及非整数小时。
2. 预测接口接受0.5等小时后将目标Time转int，可能输出错误时间：独立失败测试已复现，训练和推理现在使用同一整小时契约。
3. DL旧逻辑先写权重/图表再检查既有协议冲突：将非空输出预检前移，避免在拒绝混用前破坏历史结果。Web原有独立run_id路径兼容。多模型仍可同一次CLI调用；不同次运行必须各用新目录。
4. 当前DL训练原本已在验证前缀重拟合scaler，本次仅提取fit_training_prefix并补隔离/逆变换测试，不把已修正问题再次说成新修复。旧指标对应旧scaler；新指标单独保存。

src/models/{train_deep_learning_models,detect_kpi_anomalies,inference}.py为三个既有底层修改文件；新增src/data/hourly_contract.py、独立验证脚本和针对性测试。未改app.py、src/ui、src/services、src/core；无无关重构。

## 实际验收记录

- 初始48项测试在沙箱外通过；沙箱默认临时目录多次拒绝访问的失败日志保留，未冒充算法失败。最终57项测试在同一本机.venv环境执行通过。
- `python -m unittest discover -v`：57项通过；`python -m compileall -q src tests scripts`通过；`python -m pip check`无损坏依赖。
- AppTest在沙箱外执行六页导航全部无异常，亦无可见error；旧组件use_container_width产生弃用警告，遵照Web不改约束仅记录。
- 真实新GRU、RF四KPI、XGB四KPI共9工件：通过现有predict_kpi、与逐beam-hour保存预测比对、CSV往返；原异常工件5760条通过detect_anomalies及共识核对。
- 现有DL CLI单轮512训练/验证/测试烟测通过，仅证明功能，成绩不入新正式比较表。
- W4 p值/CI未重跑；全量标记与相关矩阵核对通过。原始报告/实验数值及Web保护哈希最终核对见preservation_check.json。

## 分阶段记录

| 阶段 | 已执行动作与原因 | 记录/产物 | 状态 |
|---|---|---|---|
| 1 全局审查 | 核对模块职责、schema/参数归属/重复、依赖与数据流；基线manifest和追踪矩阵 | docs/参数登记簿_最终审查.md、导师反馈追踪矩阵.md、parameter_inventory.csv、baseline_manifest.json | 完成 |
| 2 最小修复 | 上述失败测试驱动小时契约与输出保护；scaler隔离回归 | regression_before.txt、fractional_before.txt、regression_after.txt、git diff | 完成 |
| 3 模型实验 | 三折及两seed留出，固定参数且同训练标签、完整小时损失配对 | experiments/下配置/指标/检验/摘要，experiment_run.txt | 完成（重点三模型） |
| 4 W4证据 | 原结果复用，仅核对逻辑与矩阵 | w4_verification.json | 完成 |
| 5 报告修订 | 十份原Word的独立修订副本，包含新验证和W4整合；旧原件保留 | deliverables/review_20261025、report_revision_manifest.json、QA | 内容完成；版面以QA记录为准 |
| 6 最终验收 | 57自动测试、六页面AppTest、真实权重/导出、源码和历史保护hash | final_tests.txt等、preservation_check.json、git状态 | 自动验收完成，手工验收边界已披露 |

## 未解决风险与Git同步

本机仅CPU；无独立外部网络数据、无经核验的故障标签；HAC及30站聚类推断近似。两seed与三个时间折有限；DLThpVol大误差仍敏感；旧正式成绩不自动推广为新代码成绩。未开展额外全模型调参、生产实时接口或资源专项。

本机websockets=17.3超出requirements.txt的<17约束，pip check仍无损坏依赖（pip check不检查项目未安装清单的版本边界）。现有页面/服务测试通过，但跨机按requirements重建的等价性未验证。依赖清单与Web集成保持原样，此项列为受约束问题，不擅自升级/降级或改Web。弃用警告同样记录待后续授权处理。

只读核对远端：当前本地HEAD0781ca9与远端archive/5g-kpi-web-system-2026-10-23-rc1相同；远端main为45dfa843，当前本地分支的原上游已删除。不能再称本地当前分支与远端main完全同步。本次修改留在本地工作树，未提交、未推送、未合并或改Web分支。远端证据remote_heads.txt，最终本地状态git_status_latest.txt，文件清单changed_files.txt。未获要求的远端改写不自动进行。
'''
    (ROOT/'docs/导师反馈逐项关闭清单.md').write_text(closure,encoding='utf-8')
    index='''# 核验修订报告交付索引

2026-10-25。本目录为本轮交付版本。原Week2/Week3/Week4报告及历史CSV/图表保持归档，避免覆盖历史成绩；不要混用15页旧PDF、16页旧PDF和本次带日期修订版的页码。

优先阅读`完整项目技术报告_核验修订20261025.docx`及pdf/中的对应版本；各阶段独立修订文件均在同目录。W2统计报告由真正的数据分析Word修订，不再以错名模型报告代替。

源代码审查、阶段日志、参数登记和最终状态见项目docs/导师反馈逐项关闭清单.md。新实验附录与历史主表分开；W4统计复用已有实测结果。十份报告来源及SHA256见reports/final_audit_20261025/report_revision_manifest.json，版面检查见document_qa.json。

图表的原物理换算单位未知时使用数据集原始KPI尺度；标准化MSE与原尺度误差明确区分。新测试57项、六页AppTest通过，不将历史26项或CLI烟测成绩替代正式模型表现。
'''
    (ROOT/'deliverables/review_20261025/交付索引.md').write_text(index,encoding='utf-8')
    # Append current verification to canonical editable Markdown; leave old scored tables untouched.
    notice='''\n\n## 2026年10月25日最终核验更新

本次已完成重点GRU/RF/XGB三折扩展窗口及两组种子留出复核，修正后的训练前缀scaler与历史正式成绩分别归档。统一60,000训练标签的实际指标和配对HAC/Holm结果见[模型实验与统计复核](../reports/final_audit_20261025/模型实验与统计复核.md)，导师十项要求、受约束问题及最终验收见[逐项关闭清单](导师反馈逐项关闭清单.md)。

当前底层已拒绝缺小时/重复/非整数小时，并在DL训练写出前拒绝覆盖旧实验。原Web程序和集成服务未改，57项自动测试、六页AppTest和真实权重/CSV往返通过；完整真人上传/下载及实时训练流程未重做。本文此前的26项或新版待验收描述属于历史记录，以此日期更新和关闭清单为最新状态。

完整项目技术报告及各阶段Word的本次交付版见deliverables/review_20261025/；原件保留归档。历史GRU四项MAE领先不是修正后数据的替代；新seed42中DLThpVol对RF的两种损失均未显著，其余三项GRU对RF有显著改善。未获故障真值，不报告故障准确率或因果根因。\n'''
    for path in [ROOT/'docs/TECHNICAL_REPORT.md',ROOT/'docs/ARCHITECTURE.md']:
        s=path.read_text(encoding='utf-8')
        if '## 2026年10月25日最终核验更新' not in s:path.write_text(s+notice,encoding='utf-8')
    matrix=ROOT/'docs/导师反馈追踪矩阵.md'
    s=matrix.read_text(encoding='utf-8')
    if '## 最终状态更新' not in s:
        matrix.write_text(s+'\n\n## 最终状态更新\n\n以上保留开始修复前的基线状态。实际完成状态、每条证据及遗留限制以[导师反馈逐项关闭清单](导师反馈逐项关闭清单.md)为准。本次重点三模型验证已执行，57项测试及六页AppTest通过；不将未重做的全手工Web回归标为完成。异常DataFrame推理原有diff检查已拒绝重复/断档，本次仅补整数小时检查，不误报其已具备的校验。\n',encoding='utf-8')
    env={name:md.version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','xgboost','lightgbm','streamlit','websockets']}
    (AUDIT/'installed_versions.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    artifact_files=list((ROOT/'resources/final_audit_20261025').rglob('*.npz'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_files}
    input_path=ROOT/'data/processed/kpi_timeseries.csv'
    hashes[str(input_path.relative_to(ROOT))]=hashlib.sha256(input_path.read_bytes()).hexdigest()
    (AUDIT/'experiment_input_artifact_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    changes=subprocess.run(['git','status','--short','--untracked-files=all'],cwd=ROOT,capture_output=True,text=True).stdout
    (AUDIT/'changed_files.txt').write_text(changes,encoding='utf-8')
    print('Closure checklist, experiment report, release index and canonical docs updated')


if __name__=='__main__':main()
