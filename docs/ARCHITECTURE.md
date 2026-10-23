# 项目架构与迁移约定

## 模块职责

| 包 | 负责 | 不应负责 |
|---|---|---|
| `src.core` | KPI 字段、项目根路径、通用契约 | 训练和网页逻辑 |
| `src.data` | gotsf-ds 下载、审计、预处理、特征 | Web 日志终端 |
| `src.models` | 预测模型、训练、异常检测、推理 | Streamlit 页面状态 |
| `src.analysis` | 指标分析、统计检验、稳健性验证 | 权重注册 |
| `src.services` | 本地权重枚举归档、后台任务、WebSocket 日志 | 预测算法本身 |
| `src.ui` | Streamlit 工作流、模型推理及按版本选择的结果展示 | 模型参数更新和训练算法 |

`app.py` 是 UI 总入口。`src/` 根目录只保留 `__init__.py` 和六个功能子包，所有调用统一使用完整模块路径。

## 边界和文件

1. **神经网络结构** 在 `src/models/architectures.py` 的 `ForecastNet`。训练 `src/models/train_deep_learning_models.py`、推理 `src/models/inference.py` 使用同一类，以保持 PyTorch `state_dict` 兼容。
2. **统一字段及根目录** 定义于 `src/core/config.py`。历史大脚本里仍有局部常量，后续可通过逐个测试改用统一定义；本次仅迁移明确可靠的引用，避免大规模行为变化。
3. **WebSocket 服务**：`src/services/web_stream_worker.py` 读取运行进程 stdout/stderr，事件流推送本机页面，同时批量记录 `resources/jobs/current.log`；UI 在 `src/ui/web_workflow.py`。
4. **任务执行**：`src/services/web_jobs.py` 的任务名单是固定的。`src/core/training_config.py` 是训练可配置参数、默认值和 CLI 选项映射的唯一校验来源。Web 经 WebSocket 工作者将 JSON 参数传给任务进程，任务进程再次校验并转换为安全的参数数组（不通过 Shell 解析）。传统 ML、DL 与异常检测的 **Web 训练**均使用唯一的 `resources/experiments/<run_id>`，保存 `requested_parameters.json`，再发布权重到 `resources/models/<run_id>`。传统 ML 不带 `--out` 的旧命令行运行保留原固定路径，确保历史复现入口兼容。
5. **结果读取与版本隔离**：`src/ui/dashboard_data.py` 能按显式选中的实验 `run_id` 读取 `resources/experiments/<run_id>/reports/`，也允许读取顶层历史 `reports/`。`app.py` 为 KPI 预测/模型评估选择 ML 或 DL 实验，为异常统计/相关分析选择异常实验，并从 `st.session_state` 显示当前会话推理结果。新版本不存在的显著性检验不回退引用历史 p 值；没有真值的单次推理也不呈现 MAE/RMSE/R²。新增版本选择流程待本机回归验收。

## 输出目录及 Git

`data/raw/`、`data/processed/`、`models/`、`resources/` 默认被 Git 忽略。代码迁移不得移动、覆盖这些本地目录。仓库保存 `reports/`、`figures/` 中现有结果和源代码。真正的权重与训练记录应当从本机单独备份。
