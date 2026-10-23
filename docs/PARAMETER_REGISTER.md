# 5G KPI 项目参数登记簿（第七、八周封版审查）

> 基于 `w4/local-integration-2026-10-23` 代码结构核对。**区分配置值、可派生常量和运行时状态**；迁移建议不表示已完成修改。为避免最后一天引入额外风险，本次先登记、定位重复和给出目标归属，不在未经真实训练回归的情况下批量改变算法常量。

## 一、共享数据契约

| 参数 / 同义名称 | 含义与单位 | 当前定义位置 | 引用位置 | 重复 / 漂移风险 | 建议唯一归属 |
|---|---|---|---|---|---|
| `KPIS` / `KPI_COLUMNS` | 四个源 KPI 字段名，无单位 | `src/core/config.py`；预处理、特征、ML/DL、异常脚本亦定义 | 数据加载、训练标签、推理、UI | **重复定义**；新增/重命名 KPI 容易漏改 | `src/core/config.py:KPIS` |
| `KEYS` / `ID_COLUMNS` | `base_station, cell, beam` 组合标识符 | `src/core/config.py`；DL、异常、特征脚本另定义 | 面板对齐、异常标签、推理 | **同义异名**；顺序变化可能影响 tensor reshape | `src/core/config.py:KEYS` |
| `Z_COLS` / `Z` | 标准化 KPI 列名 `<KPI>_z` | DL 与异常脚本内按各自 `KPIS` 派生 | 模型输入、异常特征 | 可派生，无需复制常量 | `[f"{k}_z" for k in KPIS]` |
| `FILES` | gotsf-ds 四原始 CSV 文件映射 | `src/data/preprocess_kpi_data.py` | 审计/预处理；UI 读取该映射 | 数据格式契约，尚未抽出共享入口 | `src/core/dataset_config.py` 或 `src/data/datasets.py` |
| `ROOT` / `PROJECT_ROOT` | 项目根目录，Path，无单位 | `src/core/config.py`，多个数据/模型脚本重算父路径 | 输入文件、权重、图表、报告 | **重复表达**；搬迁模块后层级易错 | `src/core/config.py:ROOT` |
| `RAW_DIR` / `PROCESSED_DIR` / `REPORT_DIR` / `FIGURE_DIR` | 原始数据、处理数据、报告和图表路径 | 各个旧实验脚本内定义 | CLI 数据读写、UI 展示 | 容易形成不同的固定文件路径 | `src/core/config.py`，可派生路径 |
| `OUTPUT_FILE` / `INPUT_FILE` | 处理后时间序列、特征缓存等文件名 | `src/data/preprocess_kpi_data.py`、`build_ml_features.py` | 后续训练脚本和 Web 页面 | 同名变量在不同模块语义不一样 | 各模块的输出配置对象，勿强行同名全局化 |

## 二、时序与数据划分

| 参数 | 语义和单位 | 已确认定义 | 引用与风险 | 配置建议 |
|---|---|---|---|---|
| `TRAIN_END_TIME=672` | 训练区间右开端点，小时索引 | `src/data/preprocess_kpi_data.py` | 与 ML/DL/异常切分约束相关 | 统一至数据划分配置 |
| `TEST_START_TIME=672` | 测试区间首小时 | `src/data/build_ml_features.py`；DL/异常 CLI 各有默认 672 | **同一数值重复**；更改一处会造成不一致 | 唯一数据划分常量 |
| `validation_start=588` | 深度学习验证区间首小时 | `src/models/train_deep_learning_models.py` CLI 默认值 | 对调参/早停直接影响；不能偷换成测试数据 | 数据划分配置或实验 manifest |
| `min_target_hour=169` | 最早可预测目标小时 | DL CLI | 与窗口及季节性滞后相关 | 根据序列长度与滞后合法性推导并校验 |
| `FORECAST_HORIZON=1` | 预测提前量，小时 | `src/data/build_ml_features.py` | Web 与预测 API 当前语义都是下一小时 | `src/core/forecast_config.py` 或实验 manifest |
| `LAGS=[1,2,24,168]` | 传统 ML 历史滞后，小时 | `src/data/build_ml_features.py`；`src/models/inference.py` 仍写死相同列表 | **硬编码重复**，更改特征工程会使推理漂移 | 将实际滞后列表随 ML artifact 元数据保存 |
| `seq_len=24` | DL 输入窗口，小时 | DL CLI；`src/core/training_config.py:DL_DEFAULTS` | Web 表单与模型训练需要一致 | 训练参数 schema + 模型版本 manifest |
| `seasonal_lag=168` | DL 季节性参考滞后，小时 | DL CLI；训练配置 schema | 影响最短推理历史长度 | 训练参数 schema + checkpoint JSON |
| `max_train=60000`、`max_val=12000`、`max_test=0` | DL 各分区样本上限，条；0 测试用全部 | DL CLI；`training_config.py` | 更改采样数导致实验可比性改变 | 逐实验配置，不应列为 KPI 全局参数 |

## 三、模型超参数与随机性

| 参数 | 含义/单位 | 位置 | 约束/风险 | 建议归属 |
|---|---|---|---|---|
| `epochs=12` | 最多训练轮数 | DL CLI + `training_config.DL_DEFAULTS` | 应与 patience 联合校验 | 逐实验 `requested_parameters.json` |
| `batch_size=256` | 每次梯度更新的样本数 | DL CLI + schema | 影响显存与训练速度 | 逐实验配置 |
| `lr=0.001` | 优化器学习率 | DL CLI + schema | 对收敛敏感；浮点有限值校验 | 逐实验配置 |
| `hidden=32` | RNN/Transformer 隐藏维度 | DL CLI + schema | Transformer 当前 4 个 attention heads，必须整除 | 逐模型结构元数据 |
| `patience=3` | 早停连续未改善的轮数 | DL CLI + schema | 不应大于 epochs | 逐实验配置 |
| `device=auto` | 计算设备选择，枚举 auto/cpu/cuda | DL CLI + schema | 用户选择 CUDA 不保证 PyTorch 可用 | 用户可配置；输出运行期实际设备 |
| `seed=42` / `RANDOM_STATE=42` | 随机种子，无单位 | ML、DL、异常脚本与 schema | 多处默认值重复，算法跨库随机性不同 | 实验 manifest 及共享默认 seed |
| `if_train_sample=80000` | IF 训练抽样记录数 | 异常 CLI + schema | 改变检测模型与实验可比性 | 异常实验配置 |
| `if_contamination=0.02` | IF 训练中预计的异常比例，0–1 | 异常 CLI + schema | 不等于已验证故障率 | 异常实验配置 |
| `svm_train_sample=2500`、`svm_test_sample=12000` | OCSVM 抽样记录数 | 异常 CLI + schema | OCSVM 采样与完整测试结果不可直接等量比较 | 异常实验配置 |
| `max_events=2000` | 保存异常事件最多记录数 | 异常 CLI + schema | 会截断事件清单，但不应影响检测率统计 | 异常实验配置 |
| `ML_DEFAULTS.quick=False` | 传统 ML 调试模式开关 | schema；传统 ML CLI `--quick` | 快速运行指标不能混入正式报告 | 传统 ML 实验配置 |

## 四、系统服务参数与运行状态

| 参数 / 路径 | 含义与单位 | 所在模块 | 建议 |
|---|---|---|---|
| `HISTORY_CHARS=200000` | WebSocket 内存日志缓冲字符数 | `src/services/web_stream_worker.py` | 属于服务级配置，不应属于 ML 算法 |
| `FLUSH_BYTES=64*1024` | 日志缓冲达到该字节数时落盘 | 同上 | 服务级可配置；失败退出必须 flush |
| `FLUSH_SECONDS=2.0` | 日志最大定时间隔，秒 | 同上 | 服务级可配置；不是 Web 轮询 |
| `resources/jobs/current.json` | 最近一次任务状态 | services + UI | 运行时文件，Git 忽略；多任务扩展时应改 job_id |
| `resources/jobs/current.log` | 最近任务文本日志 | services + UI | 运行时文件，Git 忽略；避免并行覆盖 |
| `resources/experiments/<run_id>/` | 每次 DL/异常训练独立实验目录 | `src/services/web_jobs.py` | **正确的版本隔离模式**，继续保留 |
| `resources/models/<run_id>/` | 发布给推理页面的已归档权重 | `src/services/model_registry.py` 等 | 与训练配置、指标、数据集来源绑定更稳健 |
| `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1` | Python 子进程输出编码设置 | `src/services/process_env.py` | 已集中，Windows 运行时需要保留 |

## 五、审查结论与收尾处理原则

- 已经具备明确的 `core/data/models/analysis/services/ui` 分层，没有必要在收尾当天再次移动目录。
- **目前最高优先级的重复定义**：四个 KPI 名称、波束 ID 字段、时间划分边界、ML Lag 列表、项目根路径。这些已有确定的重复位置。
- 其中 `LAGS` 的训练/推理同步尤其关键，建议在后续改特征时强制从模型元数据读取；本次如未做完整 ML 重新训练验证，不要擅自改变已经用来产生历史结果的 lag 语义。
- 训练参数的 UI 默认值与真实 CLI 默认值目前同值并进行了映射，但仍是两个位置；之后可以让 CLI 读取 `training_config` 的默认值，以进一步落实唯一来源原则。
- **不要把每个参数都放到一个巨型 `config.py`**：数据契约、训练实验参数、服务传输参数与运行时任务状态应分别归属对应模块。纯功能函数保留在各功能模块内。
- 工程收尾前依照 `docs/FINAL_ACCEPTANCE.md` 运行现有测试，**不为追求整洁而在最后一天无回归地更改算法行为**。
