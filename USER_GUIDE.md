# 5G KPI 预测与异常检测系统使用手册

**运行方式：**Windows/VS Code 本机的 Python + Streamlit。

## 1. 这套系统能做什么

系统使用公开 `netop/gotsf-ds` 数据集中的四项 KPI：

| 字段 | 解释 | 注意 |
|---|---|---|
| `DLPRB` | 下行物理资源块相关指标 | 保留数据集原尺度 |
| `DLThpVol` | 下行流量体积相关指标 | 不擅自标注为 Mbps |
| `DLThpTime` | 下行传输时间相关指标 | 保留数据集原尺度 |
| `MR_number` | 测量报告数相关指标 | 保留数据集原尺度 |

主要有两条独立任务：一是用传统 ML 或 LSTM/GRU/Transformer 预测**下一小时 KPI**；二是用 3σ、IQR、Isolation Forest、One-Class SVM 标记**异常候选**，再做 KPI 关联的探索性分析。两条流程都通过 Streamlit 操作。异常候选没有故障真值标签，不能等同于已确诊的网络故障。

界面同时提供**历史正式实验、按版本选择的新实验结果和当前会话的模型推理**。结果页的侧栏「结果数据来源 / 实验版本」可显式选择对应 `run_id`。测试集曲线和评估指标来自该次训练保存的 `reports/` 子目录；单次上传 CSV 的推理没有后续真实值，因此不应显示 MAE、RMSE 或 R²。页面不会把不同版本的指标、曲线和相关性文件混在一起。新版结果来源切换功能需本机回归验证。

## 2. 运行环境与启动

打开 VS Code，将仓库根目录设为当前工作目录，保留已有 `.venv`、`data/` 和 `resources/`。PowerShell 中若已显示 `(.venv)`，可以直接用 `python`；不确定时显式调用现有环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

在终端给出的 Local URL 打开页面，通常是 `http://localhost:8501`。Streamlit 端口被占用时，以实际显示为准。

项目声明支持 Python 3.10+，但具体 PyTorch、XGBoost、LightGBM、Plotly、websockets 的可用性仍取决于本机版本。

GPU 检查：

```powershell
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

仅当 CUDA 真正可用并且训练日志显示使用 CUDA 时，才能说本次训练使用了独立 GPU。

## 3. 项目目录与哪些文件会上传 GitHub

```text
项目根目录/
├── app.py                         Streamlit 入口
├── src/
│   ├── core/                      KPI 字段、通用路径、训练参数
│   ├── data/                      下载、原始数据审计、预处理、特征工程
│   ├── models/                    传统 ML、DL、异常检测、推理
│   ├── analysis/                  模型比较、异常统计、鲁棒性检查
│   ├── services/                  后台任务、WebSocket、权重版本管理
│   └── ui/                        历史展示及工作流表单
├── tests/                         结构、参数、编码、验收数据测试
├── test_inference.py              预测和异常推理合成数据测试
├── test_model_registry.py         本地模型版本发现/归档测试
├── data/raw/                      下载后的源 CSV，Git 忽略
├── data/processed/                预处理 CSV、特征缓存，Git 忽略
├── models/                        未版本化权重，Git 忽略
├── resources/
│   ├── experiments/<run_id>/      ML、DL、异常检测的独立实验结果
│   ├── models/<run_id>/           已发布的权重及元数据
│   ├── jobs/                      任务状态、日志、WebSocket 信息
│   └── demo_inputs/              Web 验收输入 CSV
├── reports/                       已追踪的历史实验报告/CSV
├── figures/                       已追踪的历史图表
├── docs/                          技术报告、审查与验收记录
└── requirements.txt
```

**新 Clone 仓库后有代码和历史统计，却没有原始数据与模型权重，是正常现象**。它们被 `.gitignore` 排除，并非遗失。新机器需要自行下载公开数据并重训，或安全地使用本机已有可信权重。

## 4. 数据获取、审计、预处理

### 4.1 Web 方式

进入「数据管理与预处理」：界面会分别显示 `data/raw/` 中四份原始 KPI CSV 是否存在。

按实际需要操作「下载原始数据」→「审计并预处理」→「生成传统 ML 特征」。如果本地已有处理好的数据和特征文件，可跳过已有步骤；**查看 Web 页面不需要重新下载、重新训练整个数据集**。

### 4.2 命令行方式

从项目根目录运行：

```powershell
python -m src.data.download_kpi_data
python -m src.data.audit_kpi_data
python -m src.data.preprocess_kpi_data
python -m src.data.build_ml_features
```

数据源固定为 `netop/gotsf-ds` 的四项 CSV，并非任意格式 CSV 的自动适配器。主要产物：

| 操作 | 本地输入 / 输出 |
|---|---|
| 下载 | `data/raw/DLPRB_train_0w-5w.csv` 等四份源文件 |
| 审计 | `reports/raw_data_audit.json`、`reports/raw_data_audit_summary.csv` |
| 预处理 | `data/processed/kpi_timeseries.csv`、`reports/preprocessing_summary.json` |
| ML 特征 | `data/processed/ml_features.pkl`、`reports/ml_feature_summary.json` |

历史预处理报告记录 2,880 个波束、840 小时、2,419,200 条观测；训练时间 0–671，测试时间 672–839。StandardScaler 和 MinMaxScaler 参数在训练期间拟合，再应用于测试区间，防止把未来测试分布用于估计训练转换参数。

特征工程使用标准化 KPI 的时刻 `t` 与滞后 1、2、24、168 小时值，目标是 `t+1` 的 KPI 原始尺度值。不要把原始时间序列 CSV 与 ML 特征缓存混成一个文件。

## 5. 传统机器学习 KPI 预测

在「模型训练」选择「传统 ML」，确认 `data/processed/ml_features.pkl` 存在后开始训练。脚本包含 Persistence 基线、LinearRegression、DecisionTree、RandomForest、XGBoost、LightGBM。Web 默认勾选**快速验收模式**，可设置训练/测试抽样上限；取消勾选则执行完整训练和测试。它尚未开放每个算法独立的树深度、树数量等超参数界面。新版传统 ML 快速训练、模型发布和 Web 预测 CSV 导出已经本机验收通过。

命令行：

```powershell
python -m src.models.train_traditional_models --out resources/experiments/manual_ml_full
```

**Web 启动的传统 ML 训练**会把模型、指标、预测汇总和图表分别写入 `resources/experiments/<run_id>/models/`、`reports/` 与 `figures/`，再将成功模型发布到 `resources/models/<run_id>/`。因此不需要覆盖仓库已跟踪的正式历史报告。**旧命令行调用（不带 `--out`）仍采用固定的 `models/`、`reports/`、`figures/` 路径**；若自行从终端训练，请明确指定独立 `--out`。

传统 ML 本机快速验收命令示例（每次使用新目录）：

```powershell
python -m src.models.train_traditional_models --quick --max-train 512 --max-test 512 --out resources/experiments/manual_ml_smoke
```

Web 测试推荐默认 512 条训练/测试抽样。**quick 模式指标不能写入正式模型效果结论**；正式指标仍采用仓库保存的完整历史实验数据。

## 6. 深度学习 KPI 预测

在「模型训练」选 LSTM、GRU 或 Transformer。Web 可配置：

| 训练参数 | Web 初始值 | 作用 |
|---|---:|---|
| Epochs | 12 | 最大训练轮数 |
| Batch Size | 256 | 每次训练更新的样本量 |
| Learning Rate | 0.001 | 学习率 |
| Hidden Size | 32 | 隐藏层维度 |
| Patience | 3 | 验证损失连续不改善的早停轮数 |
| Sequence Length | 24 | 输入历史小时数 |
| Seasonal Lag | 168 | 季节性参考历史小时数 |
| Seed | 42 | 随机种子 |
| Device | auto | 自动选择 CPU/CUDA |

高级选项还可更改训练/验证/测试抽样上限。时间切分边界不在 Web 中开放，避免破坏实验可比性。Transformer 当前有四个注意力头，Hidden Size 必须能被 4 整除。

建议初次测试使用 Epochs 3、Batch Size 64、Patience 2，先确认后台和导出链路，再决定是否做正式训练。

命令行示例（每次换一个**全新的输出目录**）：

```powershell
python -m src.models.train_deep_learning_models --models gru --epochs 3 --batch-size 64 --patience 2 --out resources/experiments/manual_gru_smoke
```

Web 启动的每次 DL 训练会自动分配 `resources/experiments/<run_id>/`，将 `requested_parameters.json`、checkpoint、训练指标和图表保存在独立实验中，并将成功训练的 `.pt` 与同名 `.json` 复制到 `resources/models/<run_id>/` 供推理。**独立实验结果与 `reports/w3_deep_learning_metrics.csv` 的历史成绩不是同一版本。**

历史深度学习指标在 `reports/w3_deep_learning_metrics.csv`，传统 ML/DL 汇总在 `reports/week3/w3_ml_dl_all_metrics.csv`、`w3_ml_dl_comparison_report.md`。训练日志若显示 Early stopping 是正常早停，需再看最终任务状态是否 completed。

## 7. 异常检测与初步关联分析

在「模型训练」选择「异常检测」，可配置 Isolation Forest 污染率、训练抽样量、One-Class SVM 抽样、最大事件保存数量和随机种子。

脚本使用 3σ、IQR、Isolation Forest、One-Class SVM 标记候选，再按 `IF AND (3σ OR IQR)` 计算共识。历史完整测试记录中，共识候选 3,165 条，占 483,840 条测试波束小时记录的 0.65%。One-Class SVM 历史结果来自抽样子集，不应与全量测试的检测比例当成相同分母。

命令行建议输出到独立目录：

```powershell
python -m src.models.detect_kpi_anomalies --out resources/experiments/manual_anomaly_smoke
```

Web 的异常训练也会创建 `resources/experiments/<run_id>/` 并保存 `anomalies.pkl`。已有的初步相关性分析报告位于 `reports/w3_anomaly_root_cause_preliminary.md` 和 `reports/w4_anomaly_diagnostics_report.md`；统计处理实现于 `src/analysis/analyze_anomaly_statistics.py`。

**没有真实故障标签**，因此不能报告故障 Precision、Recall、F1，更不能把 KPI 相关性称为已确认的故障根因。

## 8. Streamlit 八个菜单

| 菜单 | 实际功能 |
|---|---|
| 项目总览 | 历史样本规模、保存的结果概要 |
| 数据管理与预处理 | 下载、审计、预处理和特征任务 |
| 模型训练 | 传统 ML、DL、异常训练参数和实时日志 |
| 模型管理与推理 | 本机模型版本、上传历史 CSV、下一小时预测或异常候选推理 |
| KPI 预测 | 选定实验的测试集预测曲线，或当前会话的下一小时预测结果 |
| 模型评估 | 按训练版本读取 MAE、RMSE、R²；不对缺少真值的单次推理计算指标 |
| 异常检测 | 选定异常实验的标记率与候选事件，或当前会话的异常推理结果 |
| KPI 关联分析 | 选定异常实验的候选 KPI 相关矩阵；仅当该版本存在显著性文件时显示检验结论 |

**版本切换操作：**先进入「模型训练」完成一个新实验；再进入后四个结果页面，在侧栏「结果数据来源 / 实验版本」选择该实验的 `run_id`。KPI 预测和模型评估可选 ML/DL 实验；异常检测和 KPI 关联分析可选异常实验。选择「历史正式实验」时读取仓库顶层 `reports/`。在「模型管理与推理」完成上传 CSV 的实际预测/异常检测后，切到「KPI 预测」/「异常检测」，可选择「当前会话推理」查看这一次输入的结果，且不会误用历史测试集的精度指标。**选择模型权重不等于选择训练指标版本**，各结果页以侧栏明确选中的来源为准。

## 9. 实际调用推理与下载 CSV

### 9.1 生成符合格式的输入文件

已拥有 `data/processed/kpi_timeseries.csv` 时，直接运行：

```powershell
python -m src.data.prepare_demo_inputs
```

生成两份不会影响原数据的演示输入：

- `resources/demo_inputs/forecast_history.csv`：一个波束的 169 个连续小时，适合默认传统 ML 与 DL 下一小时预测。
- `resources/demo_inputs/anomaly_history.csv`：**所有波束**的最后三个连续小时，第一小时为差分上下文，适合同一数据集训练的异常检测权重。

输入 CSV 必须包含：

```text
Time,base_station,cell,beam,DLPRB,DLThpVol,DLThpTime,MR_number
```

预测输入的各波束时间应唯一且连续，ML 需要至少 169 小时历史；DL 输入长度还取决于 checkpoint 侧车 JSON 的 `seq_len` 和 `seasonal_lag`。需要更长历史时可在演示数据脚本后附 `--forecast-hours 240` 等参数。

异常检测必须与模型训练时的全部波束 ID 和排序一致，所有波束有同一条连续时间轴，并至少带一个上下文小时。不能用仅含一个波束的预测 CSV 去测试训练于全体波束的异常模型。

### 9.2 Web 推理步骤

1. 打开「模型管理与推理」，确认下拉框已列出本机可信 `models/` 或 `resources/models/` 权重。
2. 选择传统 ML `.pkl` 或 DL `.pt`，上传 `forecast_history.csv`，点击「运行下一小时预测」。
3. 检查结果表中的目标 `Time` 以及预测 KPI 字段，点击「导出预测结果 CSV」。
4. 更换为 `anomalies.pkl`，上传 `anomaly_history.csv`，点击「运行异常检测」，查看五种方法的候选数量与比例，再导出标记 CSV。
5. 推理结果应在当前 Streamlit 会话中保留，点击下载或页面重运行不会直接清空。**切换权重或上传文件后需要重新推理**；系统按权重路径和 CSV 内容校验展示结果是否匹配。

Web 目前限制上传约 50 MB。模型的 `.pkl` 采用 Python pickle 格式，**只能加载自己信任的本地训练文件**，不应从陌生来源导入。

## 10. 实时训练终端、结果归档与常见错误

训练由 `src/services/web_jobs.py` 调度，`src/services/web_stream_worker.py` 通过本机 WebSocket 推送后台 stdout/stderr，页面为只读输出窗。状态文件 `resources/jobs/current.json`，日志 `resources/jobs/current.log`，工作器故障诊断 `resources/jobs/stream_errors.log`。关闭网页不会主动结束后台进程，但不要在训练期间删除项目或覆盖正在执行的脚本。

- **`UnicodeEncodeError: 'gbk' codec ... R²`**：Windows 进程编码问题。最新代码通过 `src/services/process_env.py` 强制 UTF-8；请确认已拉取最新分支并重新启动 Web/任务。
- **`different train_scale_mean`**：把不同标准化协议的 DL 实验混写到同一历史报告目录。Web 已为 DL 自动创建独立输出目录；CLI 必须指定一个未用过的 `--out`。
- **`ModuleNotFoundError: src.train_...`**：旧平铺式脚本已删除。改用 `python -m src.models.train_...`。
- **模型下拉框为空**：GitHub 报告 CSV 不等于可加载权重；检查本机 `models/` 或 `resources/models/`。DL checkpoint 必须有同名 JSON 元数据。
- **CUDA 不可用**：核对 `torch.cuda.is_available()` 与 PyTorch/CUDA 安装，不要仅凭 RTX 显卡存在推断训练正在使用 GPU。
- **结果仍显示旧图表**：检查后四个结果页侧栏「结果数据来源 / 实验版本」是否仍选择「历史正式实验」，切到本次 `run_id` 即可。新实验未完成或缺少对应报告文件时不会作为可选版本。
- **WebSocket 断连**：先看 `stream_errors.log` 和 `current.log`，不要反复启动相同训练任务。
- **无法导出结果**：确认先点击过对应模型的推理/异常检测按钮，下载结果必须与当前模型和上传文件匹配。

