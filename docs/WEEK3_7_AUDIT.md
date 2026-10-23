# 第 3–7 周代码与交付物核对表

**核查范围：**原八周任务书的第 3–7 周，对照工作分支 `w4/local-integration-2026-10-23`。第 1–2 周仅是此前独立的仿真与协议分析，不纳入本次脚本/仓库验收；第 8 周主要是技术报告与最终交付。

> **判定原则：**GitHub 中有代码、结果文件，不代表用户 Windows 环境的最新代码已经完成端到端测试。下表把「静态文件齐全」与「运行验收通过」分开。

## 1. 对照原任务书

| 原周次 | 原任务书交付物 | 对应代码和产物 | 静态核对 | 本机执行验收 |
|---|---|---|---|---|
| W3 开源数据与预处理 | 数据预处理 Python 脚本 | `src/data/download_kpi_data.py`、`audit_kpi_data.py`、`preprocess_kpi_data.py`、`build_ml_features.py` | 已有 | 待最后复核 |
| W3 | 预处理后 KPI CSV | `data/processed/kpi_timeseries.csv`，由预处理脚本生成 | 生成流程已有，CSV 由 Git 忽略，不在仓库 | 需本机确认文件 |
| W3 | 数据统计报告 | `reports/raw_data_audit.json`、`reports/raw_data_audit_summary.csv`、`reports/preprocessing_summary.json`、`reports/ml_feature_summary.json` | 历史记录齐全 | 不代表最新重跑 |
| W4 传统 ML 预测 | 传统预测模型代码 | `src/models/train_traditional_models.py`、`src/models/inference.py` | 已有 | Windows GBK 修复后需完整复测 |
| W4 | MAE / RMSE / R² 评估报告 | `reports/model_metrics.csv`、`reports/model_run_config.json` | 历史记录齐全 | 最新版本未重新确认 |
| W4 | 预测可视化图表 | `figures/*_predictions.png`、`figures/model_comparison_*.png`；`reports/prediction_timeseries.csv` | 历史产物齐全 | UI 页面待最终检查 |
| W5 深度学习预测 | LSTM/GRU/Transformer 代码 | `src/models/train_deep_learning_models.py`、`architectures.py` | 已有 | 新参数表单端到端待核实 |
| W5 | DL 性能报告 | `reports/w3_deep_learning_metrics.csv`、`reports/w3_deep_learning_run_config.json` | 历史记录齐全 | 新训练独立保存，不能冒充历史成绩 |
| W5 | 传统 ML 与 DL 比较 | `src/analysis/compare_model_performance.py`、`reports/week3/w3_ml_dl_all_metrics.csv`、`reports/week3/w3_ml_dl_comparison_report.md` | 历史记录齐全 | 旧指标不随下拉权重更新 |
| W6 网络异常检测 | 3σ/IQR/IsolationForest/OCSVM | `src/models/detect_kpi_anomalies.py`、`src/models/inference.py:detect_anomalies` | 已有 | Web 异常推理最近新增，待本机跑通 |
| W6 | 异常结果报告 | `reports/w3_anomaly_method_summary.csv`、`reports/w3_anomaly_candidate_events.csv`、`reports/w4_anomaly_diagnostics_report.md` | 历史记录齐全 | 未有故障真值，不能称准确率 |
| W6 | KPI 关联与初步根因方向 | `src/analysis/analyze_anomaly_statistics.py`、`reports/w3_anomaly_root_cause_preliminary.md`、`reports/w4_anomaly_kpi_correlation_tests.csv` | 已有 | 仅探索性相关分析，不是因果诊断 |
| W7 集成与 Web | 完整系统代码 | `app.py`、`src/core,data,models,analysis,services,ui`；`requirements.txt` | 分层入口齐全 | 待完整集成回归 |
| W7 | 可运行的 Web 应用 | Streamlit 八个菜单，WebSocket 日志，训练参数，版本选取，预测/异常 CSV 导出 | 本机曾启动；近期 UI 修复已提交 | 新版本待实测 |
| W7 | 系统使用手册 | 根目录 `USER_GUIDE.md`；`README.md`、`docs/ARCHITECTURE.md` | 已编写 | 需逐步按手册实测 |
| W7 | 测试与优化 | `tests/` 与根目录两个合成测试；独立实验输出、Windows UTF-8、Streamlit 持久化导出修复 | 有测试代码 | 未声称最新分支全部通过 |

## 2. 关键数据、训练和展示边界

- 原始公开数据来自 `netop/gotsf-ds` 四个固定 KPI CSV。历史统计：2,880 波束、840 小时、2,419,200 波束小时记录。时间 0–671 作为训练区间，672–839 作为测试区间；部分深度学习实验从训练期划出 588–671 作为验证期。
- ML 特征使用标准化的当前时刻及滞后 1、2、24、168 小时值，目标为下一小时的 **原尺度 KPI**。推理 CSV 每波束至少 169 连续小时；DL 要满足对应权重 JSON 中的窗口、季节性滞后要求。
- 历史比较记录 `reports/week3/w3_ml_dl_all_metrics.csv` 对全部四项 KPI 给出 MAE、RMSE、R²。Web 下拉选择新版权重时，旧 `reports/` 的历史曲线**不会自动变成该版本的重新测试指标**。
- `IsolationForest AND (3σ OR IQR)` 在完整测试集留下 3,165 条异常候选（0.65%）。One-Class SVM 的历史标记率来自 12,000 条抽样测试记录；四种方法的检测比例不是带标签的精确率。
- 预测训练与异常检测是共享 KPI 数据的两条并行实验链路。任务书没有要求必须「预测输出直接输入异常检测」。本系统分别集成两类模块并统一到 Streamlit。
- Web 的新深度学习/异常检测实验存于 `resources/experiments/<run_id>/`，发布模型于 `resources/models/<run_id>/`；传统 ML 仍沿用部分固定 `reports/` 路径，属于已知版本管理限制。

## 3. 仓库完整性的含义

**GitHub 应当提供：**可读源代码、依赖声明、数据获取/预处理命令、旧实验的报告/图表、代码测试、使用手册、技术报告、架构及参数登记。

**GitHub 不包含：**`data/raw/`、`data/processed/`、`models/`、`resources/` 中的大型原始/处理中间文件、训练出的权重、日志和本地虚拟环境。这些在 `.gitignore` 中排除，是有意的。新机器需要执行数据下载、预处理和训练命令才能完整复现实验；历史实验报告和图片本身可直接查看。

## 4. 已确认的工程风险与建议

1. **重复参数**：`KPIS/KPI_COLUMNS`、`KEYS/ID_COLUMNS`、训练/测试时间、ML 滞后窗口，在不同模块重复表达。详见 `docs/PARAMETER_REGISTER.md`。最后一天不未经回归改变模型的特征定义。
2. **最新 Windows 回归待做**：已修复 Unicode 输出及独立训练实验路径，不能拿修复提交当作完整训练已成功的证据。
3. **Web 推理结果持久化**：Streamlit 的按钮点击会使页面重新执行，下载前推理结果需要保存到 session state。当前工作分支已加入按权重路径、上传文件 SHA-256 标记的结果保存机制，仍需人工点击验收。
4. **运行时受信任文件**：`.pkl` 可能执行反序列化代码，模型管理只应选择本机可信训练产物，不提供陌生 pickle 文件上传。
5. **跨平台依赖与资源**：Python 3.14、CUDA 与机器可用内存取决于本地环境。测试报告应实测记录而不是推断。

## 5. 最终检查命令

在项目根目录的既有 Python 虚拟环境执行：

```powershell
python -m pip check
python -m compileall -q src app.py
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
python -m src.data.prepare_demo_inputs
python -m streamlit run app.py
```

本地 Web 中分别选择已保存的预测和异常模型，上传 `resources/demo_inputs/` 下的两份 CSV，**实际点击导出**；若最新训练完成，核对 `resources/jobs/current.json`、实验配置、权重及日志。本文件只记录仓库静态检查，不冒充用户本地的执行输出。
