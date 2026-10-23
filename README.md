# 5G KPI 预测与异常检测系统

基于公开 gotsf-ds KPI 时间序列数据，提供传统机器学习与深度学习的 KPI 预测、统计和机器学习异常候选检测，以及 Streamlit 交互式展示。

## 数据和模块流程

原始 KPI 数据 (data/raw/)
  → download_kpi_data.py（获取）
  → audit_kpi_data.py（检查）
  → preprocess_kpi_data.py（清洗和训练集拟合标准化）
  → data/processed/kpi_timeseries.csv

预测分支：
  处理后的 KPI 数据
  → build_ml_features.py → train_traditional_models.py
  ↘ train_deep_learning_models.py (LSTM / GRU / Transformer)
  → compare_model_performance.py → reports/、figures/

异常检测分支：
  处理后的 KPI 数据 → detect_kpi_anomalies.py
  → analyze_anomaly_statistics.py → reports/、figures/

展示分支：
  reports/ → src/dashboard_data.py (读取和字段对齐)
  → app.py (Streamlit 交互式浏览、筛选和下载)

**重要**：当前 Web 展示的是已生成的离线实验结果，不会自动训练或执行在线模型推理。原来的 free5GC 和 UERANSIM 是独立的网络协议与环境实践，尚未接入此 KPI 数据链路。异常标记仅为未经真实故障标签验证的候选事件。

## 安装

Python 3.10+，在项目根目录执行：

```bash
python -m venv .venv
pip install -r requirements.txt
```

Windows PowerShell 环境激活：` .venv\Scripts\Activate.ps1 `。
若只使用已有的 Web 报告，可以直接安装 Streamlit 和 Plotly；完整深度学习训练需要 PyTorch。GPU 版本依平台选择。

## 启动 Web

```bash
python -m streamlit run app.py
```

界面与 `src/dashboard_data.py` 会读取 `reports/` 内的现有结果，相关结果文件缺失时展示提示，不进行耗时计算。

## 实验流程（可按需要执行）

**所有脚本从仓库根目录以 Python 模块方式启动**，避免导入路径受当前工作目录影响。

```bash
# 原始数据获取、检查、预处理
python -m src.download_kpi_data
python -m src.audit_kpi_data
python -m src.preprocess_kpi_data

# 预测
python -m src.build_ml_features
python -m src.train_traditional_models
python -m src.train_deep_learning_models
python -m src.compare_model_performance

# 异常检测
python -m src.detect_kpi_anomalies
python -m src.analyze_anomaly_statistics

# 模型检验和复核（按需）
python -m src.validate_model_robustness backtest
python -m src.validate_model_robustness holdout
python -m src.review_model_results
```

可以使用 `python -m src.<module> --help` 查看多数脚本的命令行选项。完整训练和模型检验耗时较长，并非启动 Web 的必要步骤。

## 目录设计

只保留现有 `src/`（脚本）、`reports/`（实验表、参数和 Markdown）、`figures/`（图表）、`data/`（本地数据）。入口 `app.py` 位于仓库根目录，`src/dashboard_data.py` 是展示层的数据接口，防止 UI 与不同模型的输出格式强耦合。

源代码脚本使用功能化命名，不再将实际项目周次写入文件名。为保持复现实验和 Web 读取路径稳定，已有 `reports/w3_*.csv`、`reports/w4_*.csv` 和图片等结果文件名暂不变动。

原始数据与特征缓存在 `.gitignore` 里被排除。若另一个人重新克隆仓库需要复现训练，先运行数据获取和处理流程；Web 演示可直接使用版本库已收录的结果 CSV/JSON。

## 限制

- 当前为离线结果可视化，未接入实时 5G 网络采样或在线推理。
- 候选异常非已确证故障，不能在无标注数据上宣称检测的精确率/召回率。
- 不同模型的训练样本量/结构与训练策略不完全相同，跨模型比较应依据报告限定条件。

使用说明见 [USER_GUIDE.md](USER_GUIDE.md)。

第七周本地模型生成与统一预测接口说明见 [USER_GUIDE.md](USER_GUIDE.md#第七周部署接口本地生成模型)。Web 目前仍使用历史报告，尚未接入实时推理。部署接口仍需补齐异常元数据校验和完整集成验证。


## 第四阶段：交互式任务入口与版本化模型（开发中）

启动 `python -m streamlit run app.py`，新增页面：
- **数据管理与预处理**：检查 gotsf-ds 原始文件，启动固定数据集下载、审计、预处理和 ML 特征生成。
- **模型训练**：启动传统 ML、LSTM、GRU、Transformer、异常检测训练。训练由 `src.web_jobs` 子进程执行，任务结果记录在 `resources/jobs/`。
- **模型管理与推理**：扫描本地 `models/` 和 `resources/models/`，从下拉框选择权重及版本；对包含完整必要字段的历史 CSV 运行下一小时 KPI 预测，并下载结果。

`resources/models/<UTC日期时间_随机后缀>/` 是本地模型归档目录。训练前自动归档已有旧模型，训练成功后归档新文件；旧权重不被备份操作覆盖。深度学习 `.pt` 必须有对应的 `.json` 网络配置才能被列出。传统模型要求对应的 `.pkl`。权重和资源目录不应上传 GitHub。模型文件只能使用本人可信本地生成的文件。

**限制**：这些页面是离线工作流，不提供实时核心网采集、在线告警或后台任务队列。数据获取与预处理目前只支持已有 gotsf-ds 四 KPI 格式。深度学习训练需要单独安装与平台相容的 PyTorch。原有历史结果表不等于保存好的模型权重，初次部署需要下载数据并重新训练。启动按钮仅提交作业，不能代替真实环境执行验证。

检查模型归档与历史推理接口的单元测试：

```bash
python -m unittest test_model_registry test_inference -v
```

尚需真实数据的全量训练、完整 Streamlit 点击流和性能测试；不能据此声称完整交付已经验收。
