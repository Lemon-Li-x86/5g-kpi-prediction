# 系统架构与模块职责

## 1. 总体设计

系统使用 Python 与 Streamlit 构建离线 5G KPI 分析应用。数据处理、训练评估、模型版本管理、预测推理、异常检测和关联分析分别由独立模块完成，`app.py` 统一组织六个 Web 页面。

```text
公开 KPI 数据 (netop/gotsf-ds)
        │
        ▼
src.data：下载 → 审计 → 预处理 → 特征/演示输入
        │
        ├─────────────────────┐
        ▼                     ▼
预测模型训练               异常检测模型训练
ML / LSTM / GRU /          3σ / IQR / IF /
Transformer               One-Class SVM
        │                     │
        └──────────┬──────────┘
                   ▼
          src.services：实验与权重归档
          experiments/<run_id> / models/<run_id>
                   │
         ┌─────────┴───────────┐
         ▼                     ▼
  训练后测试集评估          选择权重与输入 CSV
  指标、曲线、统计         ├─ 下一小时 KPI 预测
                          └─ 异常候选检测 → KPI 关联分析
                   │
                   ▼
          src.ui / Streamlit：可视化与导出
```

这是基于公开数据集的本地训练与推理系统。free5GC / UERANSIM 网络连通性及协议抓包作为前期独立实验保留。

## 2. 模块职责

| 模块 | 职责 | 主要文件 |
|---|---|---|
| `src.core` | 统一 KPI 字段、文件路径、训练参数默认值及校验 | `config.py`、`training_config.py` |
| `src.data` | 数据下载、原始数据审计、预处理、传统 ML 特征、演示 CSV | `download_kpi_data.py`、`preprocess_kpi_data.py`、`build_ml_features.py`、`prepare_demo_inputs.py` |
| `src.models` | 传统 ML 与 DL 训练、共用网络结构、预测推理、异常检测 | `train_traditional_models.py`、`train_deep_learning_models.py`、`architectures.py`、`detect_kpi_anomalies.py`、`inference.py` |
| `src.analysis` | 模型性能比较、异常统计、稳健性检查及本次候选 KPI 关联 | `compare_model_performance.py`、`analyze_anomaly_statistics.py`、`session_kpi_analysis.py` |
| `src.services` | 后台任务执行、模型发现与归档、WebSocket 日志推送 | `web_jobs.py`、`model_registry.py`、`web_stream_worker.py` |
| `src.ui` | 交互表单、训练评估展示、实际推理页面、结果版本读取 | `web_workflow.py`、`dashboard_data.py`、`result_views.py` |
| `app.py` | Streamlit 页面初始化及六个菜单分发 | `app.py` |

模块间通过 CSV、模型文件和明确定义的调用接口连接。神经网络训练与推理统一复用 `ForecastNet`；模型参数和标准化元数据与权重配套保存。

## 3. 六个页面与实际数据流

| Web 页面 | 执行入口 | 读取的数据 | 输出或展示 |
|---|---|---|---|
| 数据管理与预处理 | `render_data_page` | `data/raw/`、`data/processed/` | 预处理时间序列、特征文件及处理状态 |
| 模型训练 | `render_training_page` | 预处理数据与训练表单 | 后台日志、模型权重、按实验保存的测试指标和图表 |
| 模型管理与实验记录 | `render_model_page` | `resources/models/`、各实验 `reports/`、历史 `reports/` | 模型版本、训练结果、正式历史模型比较 |
| KPI 预测 | `render_inference_page("forecast")` | 实际权重 + 演示或上传的 KPI CSV | 下一小时 KPI 预测值、可视化及 CSV |
| 异常检测 | `render_inference_page("anomaly")` | 实际异常权重 + 对齐的 KPI CSV | 本次候选标记、逐小时统计及 CSV |
| KPI 关联分析 | `render_current_association` 或 `render_saved_association` | 本次异常检测结果及输入 KPI，或指定实验矩阵 | 候选 KPI 相关矩阵、分析样本及导出 |

**训练测试评估**使用对应实验保存的测试集真实值和预测值，可计算 MAE、RMSE、R²。**实际 KPI 推理**使用已选权重预测输入数据后的下一小时，输出预测值。两类结果分别展示。

历史正式比较位于仓库顶层 `reports/` 和 `figures/`；新训练结果以 `run_id` 隔离归档。本次异常的 KPI 关联根据 `base_station`、`cell`、`beam`、`Time` 将检测输出与输入数据一对一匹配，再计算共识候选中的 Pearson 相关系数。样本不足时显示提示。

## 4. 参数登记与配置管理

参数统一原则：**数据契约共用、运行参数按任务保存、模型专属结构独立管理、派生值通过计算获得**。同名参数先判断语义、单位、所属模型和作用阶段，再决定是否共享。下表是最终代码审查的参数登记簿，记录定义位置、引用位置与归属。

| 参数或配置 | 语义／单位／作用范围 | 定义位置 | 主要引用位置 | 分类与处理 |
|---|---|---|---|---|
| `KPIS` | 四个 KPI 字段，保持公开数据集字段顺序 | `src/core/config.py` | 预处理、ML/DL/异常训练、推理、分析和 UI | **共享单一来源**；各训练脚本直接引用 |
| `KEYS` | 波束身份字段 `base_station/cell/beam`，不带物理单位 | `src/core/config.py` | 特征工程、DL/异常数据整形、推理、会话关联 | **共享单一来源**；表格关联包含 `Time` |
| `ROOT` 与目录常量 | 仓库根路径和 `data/`、`reports/`、`resources/` 等 | `src/core/config.py` | 数据、训练、模型归档、Web | **共享路径契约**；各实验输出目录由 `run_id` 派生 |
| `DATASET_REPO_ID`、`DATASET_FILES` | 公开数据集标识和四份原始 KPI CSV 文件名 | `src/core/config.py` | `download_kpi_data.py`、`audit_kpi_data.py`、`preprocess_kpi_data.py` | **固定数据集清单**；下载、审计、预处理共用一份映射 |
| `DATASET_TEST_START_HOUR` | 历史数据集测试区间起点，小时索引，默认 672 | `src/core/config.py` | 预处理、ML 特征、DL/异常训练与诊断的 CLI 默认值 | **共享默认边界**；研究脚本仍可显式指定 CLI 切分 |
| `ML_LAGS` | 传统 ML 输入滞后 1/2/24/168 小时 | `src/core/config.py` | `build_ml_features.py`、`inference.py` | **共享训练／推理特征契约**，顺序和含义必须一致 |
| `ML_FORECAST_HORIZON` | 传统 ML 预测目标间隔：1 小时 | `src/core/config.py` | `build_ml_features.py` | **固定任务语义**，与训练目标一致 |
| `ML_HISTORY_HOURS` | 演示输入最少连续历史小时数：`max(ML_LAGS)+1=169` | `src/core/config.py` 派生 | `prepare_demo_inputs.py`、`web_workflow.py`、`inference.py` | **派生值**，无需另存同义常量 |
| `seed` 与 `DEFAULT_RANDOM_SEED` | 随机初始化／抽样种子，无单位 | `training_config.py` 分任务；共享建议默认 42 见 `config.py` | 各模型训练和诊断 | **每个任务、每次运行独立配置**；同名不等于共享随机状态 |
| `epochs`、`batch_size`、`lr`、`patience` | DL 训练轮次、批量条数、学习率、早停轮次 | `src/core/training_config.py` 的 `DL_DEFAULTS` | Web 表单、`web_jobs.py`、DL 训练 CLI | **各模型实验独立运行参数**，与实际 `run_id` 共同保存 |
| `hidden`（LSTM/GRU） | 循环隐藏状态宽度，单位：特征维 | 每次 DL 运行参数；结构解释见 `architectures.py` | `ForecastNet` 的 RNN 层和输出头 | **模型专属语义**，同名值在 LSTM/GRU 各自独立 |
| `hidden`（Transformer） | 表示向量宽度／`d_model`，单位：特征维 | 每次 DL 运行参数；结构解释见 `architectures.py` | Transformer 输入投影、注意力层和输出头 | **模型专属语义**；需可被 4 个注意力头整除 |
| Transformer `nhead=4` | 注意力头数，模型结构超参数 | `src/models/architectures.py` | `src/core/training_config.py` 的合法性校验 | **架构常量**；本版本固定为 4，不作为统一 DL 超参数 |
| `seq_len` | 输入历史窗口，小时 | 各次 DL 运行参数，默认 24 | DL Dataset、`ForecastNet`、checkpoint 侧车元数据、推理 | **DL 实验参数**，各模型可不同 |
| `seasonal_lag` | 季节参考滞后，小时，默认 168 | 各次 DL 运行参数 | DL Dataset、checkpoint 元数据、推理 | **DL 专属特征参数**；不与传统 ML 的 168 小时滞后混同 |
| `validation_start=588` | DL 训练／验证边界（小时索引） | `train_deep_learning_models.py` CLI 默认 | DL 时间划分及 scaler 拟合 | **DL 评估协议**，独立于历史测试起点 672 |
| `max_train/max_val/max_test` | 训练、验证、测试抽样上限，样本条数 | `training_config.py` 的任务专属默认值 | ML/DL Web 训练及各自 CLI | **按任务区分**；ML `quick` 与 DL 抽样策略各自处理 |
| `device` | DL 使用 `auto/cpu/cuda`，无单位 | `DL_DEFAULTS` | DL Web 表单、训练命令 | **每次训练配置**，日志记录实际运行设备 |
| `if_contamination` | IF 预期异常比例，无量纲 | `ANOMALY_DEFAULTS` | 异常 Web 表单、训练脚本 | **异常检测专属参数** |
| `if_train_sample`、`svm_train_sample`、`svm_test_sample` | IF/SVM 抽样量，单位：记录条数 | `ANOMALY_DEFAULTS` | 异常训练及保存的运行配置 | **分算法独立样本数**；SVM 测试分母单独说明 |
| `max_events` | 最多存储的异常候选记录数，单位：条 | `ANOMALY_DEFAULTS` | 异常训练报告导出 | **存储上限**，不参与候选判定逻辑 |
| 标准化 `mean/scale`、波束顺序 | 各模型实际训练拟合的变换参数及输入排列 | ML `.pkl` bundle、DL `.pt.json`、异常 `anomalies.pkl` | 各自模型的 `inference.py` | **随模型保存的状态**；不同实验不可用同名参数互相覆盖 |
| `run_id` 与 `requested_parameters.json` | 一轮独立训练的身份、用户配置和归档来源 | `src/services/web_jobs.py` | 模型版本登记、训练评估与 Web | **每轮唯一标识**，历史指标保持独立 |

**重复定义检查结果：**四项 KPI、波束标识、公开数据集文件清单、传统 ML 滞后与最短历史长度、数据集测试起点已统一。预处理和特征工程缓存还将公共配置文件加入依赖检查，确保配置更新后重新验证生成文件。各 CLI 保留与 Web 相匹配的默认参数作为独立命令行接口；Web 运行会显式传入经校验的全部参数。LSTM、GRU、Transformer 保留各自的 `hidden` 语义与模型权重，SVM/IF 的抽样分母分别记录。固定的 Transformer 注意力头数属于模型结构，若未来开放调整，应同步调整结构定义、参数校验、checkpoint 元数据及测试。

## 4.1 注释及代码维护规范

- 每个可执行脚本应有模块级说明：用途、数据入口、关键输出和运行方式；公开 API/非显然函数说明输入、输出、尺度及失败条件。
- **解释原因和约束**：时间切分、scaler 拟合范围、特征窗口、波束对齐、候选标记规则及输出隔离。简单赋值、循环和列表构造使用清晰命名即可，不逐行翻译代码。
- Python 类型与数据结构要明确：时间列 `Time` 为小时索引，KPI 为原始数据尺度或 `_z` 标准化尺度，关联键为 `KEYS + ["Time"]`；相关矩阵来自本次候选样本。
- UI 负责交互和展示，算法留在 `src.models`，纯数据计算和统计留在 `src.analysis`。保留历史脚本的业务计算约定，减少大面积格式性重排。
- 模型训练、推理和实验展示中的“同名”参数先按所属模型、单位、应用阶段识别；新增参数同时维护默认值、校验规则、CLI 传递、保存元数据和测试。
- 单元测试使用合成输入；完整 Web 点击与实际模型权重回载在目标 Windows 环境验收。新增算法输出应有独立路径并保持正式历史实验文件可复核。

## 5. 后台训练及数据归档

1. Streamlit 表单收集参数并交由 `normalize_params` 校验。
2. `src/services/web_stream_worker.py` 启动受控的 `web_jobs.py` 后台任务，通过本机 WebSocket 发送日志，持久化到 `resources/jobs/`。
3. 每次训练创建独立 `resources/experiments/<run_id>/`，保存请求参数、测试结果、图表及训练模型。
4. 训练完成后将可信权重发布至 `resources/models/<run_id>/`，供预测与异常检测页面选择。
5. 训练界面读取刚完成的实验报告，模型管理页面支持查看已有实验及历史正式结果。

典型实验目录：

```text
resources/
├── experiments/
│   └── <run_id>/
│       ├── requested_parameters.json
│       ├── models/
│       ├── reports/
│       └── figures/
├── models/
│   └── <run_id>/
├── demo_inputs/
└── jobs/
```

命令行训练使用独立 `--out` 目录可获得相同的产物隔离。历史传统 ML 命令省略 `--out` 时仍采用固定输出路径。

## 6. 源码、实验结果与版本管理

Git 跟踪源码、测试、文档，以及 `reports/`、`figures/` 中的正式历史实验成果。本地下载数据和新实验产物分别存放在 `data/`、`models/`、`resources/`，由 `.gitignore` 排除。模型权重和独立实验记录在迁移设备时应一并备份。

**验收状态（2026-10-23）**：旧版 Web 的训练、权重发布、模型推理和 CSV 导出已完成本机功能验收。六模块整合与新增关联分析测试已提交工作分支，待在目标 Windows 环境执行回归测试。


## 2026年10月25日最终核验更新

本次已完成重点GRU/RF/XGB三折扩展窗口及两组种子留出复核，修正后的训练前缀scaler与历史正式成绩分别归档。统一60,000训练标签的实际指标和配对HAC/Holm结果见[模型实验与统计复核](../reports/final_audit_20261025/模型实验与统计复核.md)，导师十项要求、受约束问题及最终验收见[逐项关闭清单](导师反馈逐项关闭清单.md)。

当前底层已拒绝缺小时/重复/非整数小时，并在DL训练写出前拒绝覆盖旧实验。原Web程序和集成服务未改，57项自动测试、六页AppTest和真实权重/CSV往返通过；完整真人上传/下载及实时训练流程未重做。本文此前的26项或新版待验收描述属于历史记录，以此日期更新和关闭清单为最新状态。

完整项目技术报告及各阶段Word的本次交付版见deliverables/review_20261025/；原件保留归档。历史GRU四项MAE领先不是修正后数据的替代；新seed42中DLThpVol对RF的两种损失均未显著，其余三项GRU对RF有显著改善。未获故障真值，不报告故障准确率或因果根因。


## 文件交付整理补充（2026-10-25）

CLI 可调默认值现引用 core/training_config 的现有字典，数值保持一致；本机路径和原研究协议保留。实验文本/图片及全部阶段文档已纳入交付归档，目录策略和参数重构边界见 [文件审计](DELIVERY_AUDIT.md)。历史哈希记录保留用于追溯，不能当作整理后源码的哈希。
