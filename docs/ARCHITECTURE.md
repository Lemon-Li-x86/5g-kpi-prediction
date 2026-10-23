# 项目架构与迁移约定

## 模块职责

| 包 | 负责 | 不应负责 |
|---|---|---|
| `src.core` | KPI 字段、项目根路径、通用契约 | 训练和网页逻辑 |
| `src.data` | gotsf-ds 下载、审计、预处理、特征 | Web 日志终端 |
| `src.models` | 预测模型、训练、异常检测、推理 | Streamlit 页面状态 |
| `src.analysis` | 指标分析、统计检验、稳健性验证 | 权重注册 |
| `src.services` | 本地权重枚举归档、后台任务、WebSocket 日志 | 预测算法本身 |
| `src.ui` | Streamlit 交互、历史图表适配 | 模型参数更新和训练算法 |

`app.py` 是 UI 总入口。保留原 `src/*.py` 薄包装层支持旧模块名，包括 `python -m src.train_deep_learning_models` 和 `from src.inference import predict_kpi`。**这些兼容文件不是另一个业务实现**，新功能请在分层目录内修改。

## 边界和文件

1. **神经网络结构** 在 `src/models/architectures.py` 的 `ForecastNet`。训练 `src/models/train_deep_learning_models.py`、推理 `src/models/inference.py` 使用同一类，以保持 PyTorch `state_dict` 兼容。
2. **统一字段及根目录** 定义于 `src/core/config.py`。历史大脚本里仍有局部常量，后续可通过逐个测试改用统一定义；本次仅迁移明确可靠的引用，避免大规模行为变化。
3. **WebSocket 服务**：`src/services/web_stream_worker.py` 读取运行进程 stdout/stderr，事件流推送本机页面，同时批量记录 `resources/jobs/current.log`；UI 在 `src/ui/web_workflow.py`。
4. **任务执行**：`src/services/web_jobs.py` 的任务名单是固定的。DL 训练输出放在唯一的 `resources/experiments/<run_id>`，然后发布权重到 `resources/models/<run_id>`。旧版 ML/异常脚本仍写到固定报告路径。
5. **结果读取**：`src/ui/dashboard_data.py` 只读取已生成的历史 CSV/JSON；不得把它展示的旧指标标为当前选定权重的性能。

## 输出目录及 Git

`data/raw/`、`data/processed/`、`models/`、`resources/` 默认被 Git 忽略。代码迁移不得移动、覆盖这些本地目录。仓库保存 `reports/`、`figures/` 中现有结果和源代码。真正的权重与训练记录应当从本机单独备份。

## 修改和测试规则

- 修改实现优先编辑 `src/<对应分层>/xxx.py`，不要在薄包装层重写逻辑。
- 保持兼容入口直到 CLI、Streamlit、推理、测试都经过真实数据回归。
- 修改模型结构时，必须测试旧 checkpoint 加载及推理输出维度；修改标准化时，不得将不同协议的指标合并。
- 修改后台任务时，测试断开浏览器、失败处理、训练结果独立归档，以及日志持续写入。
- 运行：`python -m unittest discover -s tests -v`、`python -m unittest test_inference test_model_registry -v`、`python -m compileall -q src app.py`。
- 如果训练正在执行，不要替换它正在运行的代码或在训练结束前覆盖权重与目录；代码同步应在任务结束后进行。

本次为不破坏功能的目录重构，保留历史实验方法和权重格式；尚未声称完成 Windows 端到端验收。
