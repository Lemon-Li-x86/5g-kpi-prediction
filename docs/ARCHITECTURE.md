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

## 4. 参数与共享约定

| 参数 / 字段 | 唯一来源与传递方式 |
|---|---|
| 四项 KPI、波束标识及根目录 | `src/core/config.py` 中的 `KPIS`、`KEYS`、`ROOT` |
| Web 可配置的训练参数、默认值和 CLI 映射 | `src/core/training_config.py` 中的 `DL_DEFAULTS`、`ANOMALY_DEFAULTS`、`ML_DEFAULTS` 及校验函数 |
| Web 单次训练提交的实际参数 | `resources/experiments/<run_id>/requested_parameters.json` |
| ML 模型特征、归一化元数据 | 模型 `.pkl` bundle 中的列顺序、目标与标准化参数 |
| DL 序列长度、季节滞后和归一化参数 | 与 `.pt` 同名的 `.json` 元数据 |
| 异常模型的训练波束集合、缩放与检测器 | `anomalies.pkl` 模型 bundle |
| UI 运行中的预测/异常结果 | 当前 Streamlit 会话状态，按权重路径和输入 CSV 哈希匹配 |

历史脚本中的局部实验常量保持原有计算语义。新增功能优先引用 `src.core` 的共享定义，业务逻辑及页面代码分别保留在所属模块内。

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
