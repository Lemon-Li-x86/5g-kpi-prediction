# 5G KPI 预测与异常检测系统

基于 gotsf-ds 公开 KPI 数据的离线实验系统，支持数据下载/预处理、传统 ML 与深度学习预测、异常候选检测、实验分析，以及本机 Streamlit 交互操作。Web 训练支持本地 WebSocket 日志流。

## 快速运行（从项目根目录执行）

在已有虚拟环境内安装缺少的依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

也可激活虚拟环境后使用 `python -m streamlit run app.py`。程序需要 Python 3.10+，使用 PyTorch 训练时须安装与本机 CPU/CUDA 平台兼容的版本。

## 当前功能

- **数据管理与预处理**：识别固定 gotsf-ds 四 KPI 文件；下载、审计、预处理及传统 ML 特征工程。
- **模型训练**：在 Web 表单直接调整 LSTM/GRU/Transformer 的 Epoch、Batch Size、学习率、隐藏维度、时间窗口、早停、抽样上限、随机种子及计算设备；异常检测可设置污染率和抽样参数。后台按白名单校验，再由独立进程执行。WebSocket 实时展示输出，日志同时写入 `resources/jobs/current.log`。传统 ML 可在 Web 选择快速抽样验收或完整训练，自动隔离实验目录与权重；尚不开放各算法独立的树深度等超参数。
- **模型管理与推理**：从本机 `models/` 和 `resources/models/` 枚举可信模型版本，使用历史 CSV 执行下一小时预测或异常候选标记并导出 CSV；传统 ML、GRU 与异常模型已完成各自本机推理及导出验收。
- **结果展示**：总览、KPI 预测曲线、模型评估、异常候选统计、KPI 关联分析。**后四个结果页面可明确选择历史正式实验或某个新训练的 `run_id`，按版本读取指标、曲线、异常率和相关矩阵；预测、异常页面也能选择当前会话的实际推理结果。新版版本切换功能仍待本机回归测试。**

固定支持 KPI：`DLPRB`、`DLThpVol`、`DLThpTime`、`MR_number`。异常标记是未经故障标签验证的候选事件，不应宣称真实故障检测 precision / recall / F1。

## 目录结构

```text
5g-kpi-prediction/
├── app.py                     # Streamlit 入口
├── src/
│   ├── core/                  # 统一路径及 KPI 字段
│   ├── data/                  # 下载/审计/预处理/特征
│   ├── models/                # 预测训练/异常检测/推理/神经网络结构
│   ├── analysis/              # 评估/相关性/鲁棒性
│   ├── services/              # 任务调度/WebSocket 日志/权重注册
│   ├── ui/                    # Streamlit 工作流和结果读取器
│   └── __init__.py            # Python 包标记
├── tests/                     # 结构契约测试
├── test_inference.py          # 原合成样本推理测试
├── test_model_registry.py     # 原权重归档测试
├── data/                      # 本地下载数据（Git 忽略）
├── models/                    # 旧版本地模型权重（Git 忽略）
├── resources/                 # 本地模型版本/实验/任务日志（Git 忽略）
├── reports/                   # 历史已保存的实验指标
├── figures/                   # 历史已保存的实验图
├── docs/ARCHITECTURE.md
└── USER_GUIDE.md
```

## 命令行复现实验

命令行使用新分层模块路径。

```bash
python -m src.data.download_kpi_data
python -m src.data.audit_kpi_data
python -m src.data.preprocess_kpi_data
python -m src.data.build_ml_features
python -m src.models.train_traditional_models --out resources/experiments/manual_ml_full
python -m src.models.train_deep_learning_models --models gru --out resources/experiments/manual_gru
python -m src.analysis.compare_model_performance
python -m src.models.detect_kpi_anomalies
python -m src.analysis.analyze_anomaly_statistics
```

注意：每次 Web 传统 ML、深度学习和异常检测训练都会自动分配唯一的 `resources/experiments/<run_id>/` 输出目录，并保存 `requested_parameters.json`；成功后将可用模型发布到 `resources/models/<run_id>/`。命令行复现实验也建议指定新的 `--out` 目录。**直接运行未带 `--out` 的传统 ML 命令仍沿用历史固定输出路径**，可能覆盖已跟踪报告。传统 ML Web 快速模式已于 2026-10-23 完成训练、20 个模型发布及推理导出验收。

## 检查

```bash
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
python -m compileall -q src app.py
```

仓库不包含本地训练数据或权重。代码与文档已统一使用新模块路径，但本次删旧入口后仍需在本机重新执行单元测试和 Web 点击回归。

## 文档

详细操作、模型版本、实时日志和问题排查： [USER_GUIDE.md](USER_GUIDE.md)。
技术结构： [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 一键生成 Web 验收输入

已有预处理数据后，可运行：

```bash
python -m src.data.prepare_demo_inputs
```

它将分别生成 `resources/demo_inputs/forecast_history.csv`（一个波束、169 小时，用于预测推理）和 `resources/demo_inputs/anomaly_history.csv`（所有波束、最后 3 小时，用于异常推理），不会触碰已有权重或报告。异常模型要求与训练时波束集合一致；若选择了超过 168 小时季节性滞后的 DL 权重，可通过 `--forecast-hours` 增加预测历史长度。

## 当前交付说明

本仓库已跟踪源代码、Python 单元测试、历史 `reports/` 和 `figures/`、使用手册与技术报告；大型公开源 CSV、处理中间数据及模型权重按照 `.gitignore` 留在用户本机。
