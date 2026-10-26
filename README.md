# 基于 AI 的 5G KPI 预测与异常检测系统

系统使用公开的 `netop/gotsf-ds` 波束级数据，处理 `DLPRB`、`DLThpVol`、`DLThpTime`、`MR_number` 四项 KPI，提供数据预处理、ML/DL 训练与测试、下一小时预测、异常候选筛查及 KPI 关联分析。Web 应用基于 Streamlit，在本机运行。

## 1. 安装和启动

先安装 Git LFS，再克隆仓库并获取交付数据。已有仓库直接执行 `git lfs install --local` 和 `git lfs pull`。

```powershell
git lfs install
git clone https://github.com/Lemon-Li-x86/5g-kpi-prediction.git
Set-Location "5g-kpi-prediction"
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

项目推荐 Python 3.10+。PyTorch 的 CPU/CUDA 安装版本由本机环境决定。打开启动日志中的本机 URL，默认通常为 `http://localhost:8501`。

## 2. 三份项目说明

| 文档 | 内容 |
|---|---|
| [README](README.md) | 项目介绍、安装、交付入口与验证命令 |
| [USERGUIDE](USERGUIDE.md) | 数据准备、模型训练、Web 操作、CSV 格式与排错 |
| [ARCHITECTURE](ARCHITECTURE.md) | 模块职责、数据流、参数来源及实验协议 |

项目只维护上述三份 Markdown。正式阶段报告、技术报告、使用手册和演示文稿以 `deliverables` 中的 PDF/PPT/TXT 为交付版本；实验分析保存数值 CSV、JSON 和图表。

## 3. 功能和项目结构

| Web 页面 | 功能 |
|---|---|
| 数据管理与预处理 | 下载公开原始数据、审计、清洗及构建 ML 特征 |
| 模型训练 | 参数设置、后台训练、日志、测试指标和独立实验归档 |
| 模型管理与实验记录 | 权重发现、模型版本及历史/独立实验评估 |
| KPI 预测 | 使用演示或上传 CSV，选择真实权重，预测下一小时并导出 |
| 异常检测 | 统计方法、Isolation Forest、One-Class SVM 与共识候选 |
| KPI 关联分析 | 对本次候选或指定历史实验计算和展示 KPI 相关性 |

```text
项目根目录/
├── README.md / USERGUIDE.md / ARCHITECTURE.md
├── app.py                    六页面入口
├── src/
│   ├── core/                 KPI、路径、训练参数与校验
│   ├── data/                 下载、审计、预处理、特征、演示数据
│   ├── models/               训练、网络结构、预测与异常推理
│   ├── analysis/             模型比较、统计及会话关联
│   ├── services/             模型注册、后台任务与实时日志
│   └── ui/                   表单、结果读取和展示
├── tests/                    单元和数据/结构契约测试
├── scripts/                  数据、模型及Web验证工具
├── deliverables/             四阶段交付、LFS数据集与相对路径索引
├── reports/ / figures/       正式历史指标、配置和图表
├── data/                     本地运行数据及特征缓存
├── models/ / resources/      本地权重、独立实验、演示数据和任务状态
└── requirements*.txt         运行/可选研究依赖
```

预测模型包括 Persistence 基线、LinearRegression、DecisionTree、RandomForest、XGBoost、LightGBM、LSTM、GRU 和 Transformer。异常共识采用 `IF AND (3σ OR IQR)`。没有经过核实的故障标签，候选标记与相关性用于后续排查。

## 4. 数据集与 Git LFS

交付数据集 [2_kpi_timeseries.csv](deliverables/submit2/2_kpi_timeseries.csv) 通过 Git LFS 跟踪，包含 2,419,200 行、17 列，覆盖 2,880 个波束和 840 个连续小时。普通 Git 提交保存 LFS 指针，LFS 服务保存完整 CSV；需安装 Git LFS 并执行 `git lfs pull`。

准备应用使用的副本和特征：

```powershell
git lfs pull --include="deliverables/submit2/2_kpi_timeseries.csv"
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null
Copy-Item -LiteralPath "deliverables/submit2/2_kpi_timeseries.csv" -Destination "data/processed/kpi_timeseries.csv"
.\.venv\Scripts\python.exe -m src.data.build_ml_features --force
```

该复制命令适用于首次准备运行数据；已有研究数据应按实验版本管理。交付 CSV 与旧本机处理副本的原始 KPI、身份字段和时间划分相同，标准化列存在最大约 `1e-5` 的差异，复现时应明确使用哪份数据。交付数据的 LFS 对象 ID 记录其完整内容哈希。

`data/` 下的原始数据、运行副本及特征缓存，`models/`、`resources/` 下的权重和运行结果，以及虚拟环境和缓存仍由 Git 忽略。新机器可用交付数据训练权重；完整原始数据重建步骤见 [USERGUIDE](USERGUIDE.md)。

## 5. 训练、比较与验证

每次训练使用新的 `--out` 目录：

```powershell
python -m src.models.train_traditional_models --out resources/experiments/manual_ml
python -m src.models.train_deep_learning_models --models gru --out resources/experiments/manual_gru
python -m src.models.detect_kpi_anomalies --out resources/experiments/manual_anomaly
python -m src.data.prepare_demo_inputs
python -m unittest discover -v
```

模型比较继续输出 CSV 与图表，异常分析继续输出检测统计、候选记录、相关矩阵及配置；命令不再生成额外 Markdown 报告。

2026-10-25 的本机运行核验覆盖代表性实际权重、CSV 往返、六页 Web 及当前异常关联。它不等于全量重训、现场 5G 网络复测或压力测试。本次整理后 58 项自动测试全部通过，模型比较命令的 CSV 与历史结果一致，且不再生成额外 Markdown。

历史 ML/DL 指标与严格训练前缀 scaler 复核分别保存在 `reports/` 对应子目录。历史 LSTM/GRU/Transformer 结果不得冒充新协议结果；新研究也没有提供已确认故障标签。

## 6. 正式交付入口

- [24 项交付物与相对路径对照](deliverables/项目交付物与文件路径对照.txt)
- [第一阶段：环境、协议与抓包](deliverables/submit1/)
- [第二阶段：数据、传统 ML、图表](deliverables/submit2/)
- [第三阶段：深度学习、异常检测与初步分析](deliverables/submit3/)
- [第四阶段：项目技术报告](deliverables/submit4/项目技术报告.pdf)
- [第四阶段：系统使用手册 PDF](deliverables/submit4/3_系统使用手册.pdf)
- [第四阶段：项目演示 PPT](deliverables/submit4/ppt.pptx)

正式 PDF/PPT 已于 2026-10-25 同步最终实验与验收说明。第三阶段报告纳入 GRU 时间回测、双种子与配对检验，以及异常分布、联合规则和站级关联统计；LSTM/Transformer 的架构探索记录注明独立实验条件。安装、LFS 数据获取与代码入口见三份项目说明，最新 58 项回归记录位于 `reports/delivery_final_20261025/tests.txt`。
