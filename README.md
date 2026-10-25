# 基于 AI 的 5G KPI 预测与异常检测系统

本项目使用公开的 `netop/gotsf-ds` 波束级 KPI 数据，实现数据处理、模型训练与评估、下一小时 KPI 预测、异常候选检测及 KPI 关联分析，提供基于 Streamlit 的本地 Web 应用。

系统处理四项指标：`DLPRB`、`DLThpVol`、`DLThpTime`、`MR_number`。原始字段单位与数值尺度按照数据集保留。

## 1. 安装与启动

在仓库根目录运行（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

若已激活虚拟环境，可使用 `python -m streamlit run app.py`。推荐 Python 3.10+，并根据本机 CPU/CUDA 环境安装兼容的 PyTorch。打开终端中显示的 Streamlit Local URL，通常为 `http://localhost:8501`。

## 2. Web 功能模块

| 页面 | 功能 |
|---|---|
| **数据管理与预处理** | 下载四项原始 KPI、审计与预处理、保存时间序列、构建传统 ML 特征 |
| **模型训练** | 选择预测或异常检测算法，调整训练参数，查看 WebSocket 实时日志；完成后直接查看该实验的测试指标、曲线或异常统计 |
| **模型管理与实验记录** | 管理本地模型权重，按 `run_id` 查询训练实验，查看固定的历史正式模型性能对比 |
| **KPI 预测** | 从预处理数据生成演示 CSV 或上传 CSV，选择 ML/DL 权重，计算下一小时 KPI 并可视化、导出 |
| **异常检测** | 生成或上传符合模型要求的 KPI CSV，选择异常检测权重，展示异常候选标记率、逐小时统计并导出 |
| **KPI 关联分析** | 根据本次异常检测的候选标记与输入 KPI 计算相关矩阵，也可查看某次实验保存的关联结果 |

传统预测算法包括 LinearRegression、DecisionTree、RandomForest、XGBoost、LightGBM，深度学习算法包括 LSTM、GRU、Transformer，并提供 Persistence 基线。异常检测综合使用 3σ、IQR、Isolation Forest 和 One-Class SVM。

训练评估基于测试集，展示 MAE、RMSE、R²，以及真实值和预测值的对比曲线。KPI 预测页面对实际输入执行下一小时推理，显示预测值与图表。异常检测结果用于筛选后续分析的候选记录。

## 3. 数据与模型的运行流程

```text
公开 gotsf-ds 数据
      │ 下载、审计、预处理、特征构建
      ▼
本地数据集与特征
      │
      ├── 选择算法、配置参数、训练及测试
      │        ├── resources/experiments/<run_id>/  测试指标、曲线、训练配置
      │        └── resources/models/<run_id>/       已发布模型权重
      │
      ├── KPI 预测：演示/上传 CSV + 选择权重 → 下一小时预测 → 图表/CSV
      │
      └── 异常检测：演示/上传 CSV + 选择权重 → 异常标记 → 图表/CSV
                                                       │
                                                       └── 本次候选 KPI 关联分析
```

每轮 Web 训练使用独立 `run_id` 保存参数、报告、图表及权重。此前完成的正式模型比较保存在仓库顶层 `reports/`、`figures/`；新实验保存到 `resources/experiments/<run_id>/`，两者各自独立。

传统 ML Web 提供抽样快速验收模式和完整训练模式。快速模式用于功能测试，正式预测性能比较采用对应完整实验的测试集结果。异常检测数据暂无经过验证的故障标签，当前系统展示候选标记率和探索性 KPI 关联。

## 4. 项目结构

```text
5g-kpi-prediction/
├── app.py                            Streamlit 六模块入口
├── src/
│   ├── core/                         KPI 字段、路径及训练参数
│   ├── data/                         数据下载、审计、预处理、特征与演示 CSV
│   ├── models/                       ML/DL 训练、模型结构与异常/预测推理
│   ├── analysis/                     指标比较、统计分析与本次候选 KPI 关联
│   ├── services/                     模型注册、后台训练与 WebSocket 日志
│   └── ui/
│       ├── web_workflow.py           训练表单、模型管理和实际推理
│       ├── dashboard_data.py         按实验版本读取保存结果
│       └── result_views.py           训练评估、推理与关联结果可视化
├── tests/                            单元测试与结构/数据契约测试
├── test_inference.py                 模型推理合成数据测试
├── test_model_registry.py            模型发现和归档测试
├── data/                             本地原始与处理后 KPI 数据
├── models/                           历史未版本化本地权重
├── resources/
│   ├── experiments/<run_id>/         每次训练的配置、报告、图表、模型
│   ├── models/<run_id>/              发布给推理页面的模型版本
│   ├── demo_inputs/                  Web 演示输入
│   └── jobs/                         后台任务状态和日志
├── reports/                          保存的正式历史实验结果
├── figures/                          保存的正式历史实验图表
├── USER_GUIDE.md                     系统使用手册
└── docs/
    ├── ARCHITECTURE.md               系统架构和模块职责
    └── TECHNICAL_REPORT.md           项目技术报告
```

`data/`、`models/` 和 `resources/` 为本地运行目录，默认受 `.gitignore` 保护。新机器部署时需要准备数据和可信权重；训练数据可通过 Web 下载并重新处理。

## 5. 命令行执行

```powershell
python -m src.data.download_kpi_data
python -m src.data.audit_kpi_data
python -m src.data.preprocess_kpi_data
python -m src.data.build_ml_features

python -m src.models.train_traditional_models --out resources/experiments/manual_ml
python -m src.models.train_deep_learning_models --models gru --out resources/experiments/manual_gru
python -m src.models.detect_kpi_anomalies --out resources/experiments/manual_anomaly
```

命令行训练应为每次实验指定新的 `--out` 目录。传统 ML 的旧命令行入口在省略 `--out` 时仍使用固定的历史输出路径。

完成预处理后，也可通过命令行生成 Web 演示输入：

```powershell
python -m src.data.prepare_demo_inputs
```

该命令生成 `resources/demo_inputs/forecast_history.csv`（单波束连续 169 小时）和 `resources/demo_inputs/anomaly_history.csv`（全波束连续 3 小时）。新版 Web 的 KPI 预测、异常检测页面也可直接生成这两份输入。

## 6. 测试与交付状态

```powershell
python -m compileall -q src app.py
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
```

2026-10-23 已完成原版 Web 中传统 ML、GRU 和异常模型的本机训练、权重发布、推理及 CSV 导出验收。随后完成六模块界面整理，将训练评估与实际推理分离，并新增按模型实验版本读取报告、当前异常候选关联和对应测试。**六模块新版界面的后续自动验收见 2026-10-25 最终审查记录；本次文件整理后的验证见下方交付入口。**

## 7. 相关文档

- [系统使用手册](USER_GUIDE.md)：逐步操作、演示数据及常见问题
- [系统架构](docs/ARCHITECTURE.md)：模块职责、参数归属、数据流及产物管理
- [技术报告](docs/TECHNICAL_REPORT.md)：实验方法、正式结果、Web 集成和验收记录

## 8. 2026-10-25 仓库交付入口

- [文件审计、隔离清单与提交边界](docs/DELIVERY_AUDIT.md)
- [数据获取、环境版本及模型迁移](docs/DATA_AND_MODELS.md)
- [修订后的阶段 Word/PDF](deliverables/review_20261025/)
- [原始报告输入](docs/source_reports/)与 [实验文本/图片归档](reports/experiment_archive/)
- [本轮审计及验证](reports/delivery_audit_20261025/)

运行目录 resources 继续受忽略规则保护；本次补充的实验归档副本随 Git 跟踪。原始数据、历史权重和逐样本配对预测留在本机完整备份中，模型外部发布地址尚待确认。主程序依赖位于 requirements.txt；可选调参和文档工具依赖位于 requirements-research.txt。本次不重新训练模型，不用新结果覆盖历史对比。
