# 5G KPI 预测与异常检测系统

基于 gotsf-ds 公开 KPI 数据的离线实验系统，支持数据下载/预处理、传统 ML 与深度学习预测、异常候选检测、实验分析，以及本机 Streamlit 交互操作。Web 训练支持本地 WebSocket 日志流；**不等于对接实时 5G 网络**。

## 快速运行（从项目根目录执行）

在已有虚拟环境内安装缺少的依赖，无需每次重新建环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

也可激活虚拟环境后使用 `python -m streamlit run app.py`。程序需要 Python 3.10+，使用 PyTorch 训练时须安装与本机 CPU/CUDA 平台兼容的版本。

## 当前功能

- **数据管理与预处理**：识别固定 gotsf-ds 四 KPI 文件；下载、审计、预处理及传统 ML 特征工程。
- **模型训练**：在 Web 表单直接调整 LSTM/GRU/Transformer 的 Epoch、Batch Size、学习率、隐藏维度、时间窗口、早停、抽样上限、随机种子及计算设备；异常检测可设置污染率和抽样参数。后台按白名单校验，再由独立进程执行。WebSocket 实时展示输出，日志同时写入 `resources/jobs/current.log`。传统 ML 尚无独立的超参数 CLI，Web 保留固定的正式训练配置。
- **模型管理与推理**：从本机 `models/` 和 `resources/models/` 枚举模型版本，选择可信本地权重，对符合字段要求的历史 CSV 执行下一小时预测，或选择保存的异常模型执行候选标记，并导出 CSV；本轮新增的异常 Web 推理仍须本机实测。
- **结果展示**：总览、KPI 预测曲线、模型评估、异常候选统计、KPI 关联分析。**这些历史图表读取 `reports/` 中的已有实验结果，不一定对应当前下拉框选中的模型版本。**

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

`src/` 根目录只保留 `__init__.py` 和六个功能子包，不再维护旧版扁平脚本入口。新功能直接添加到对应子包。参见 [架构说明](docs/ARCHITECTURE.md)。

## 命令行复现实验

命令行使用新分层模块路径。旧版 `python -m src.<脚本名>` **不再支持**：

```bash
python -m src.data.download_kpi_data
python -m src.data.audit_kpi_data
python -m src.data.preprocess_kpi_data
python -m src.data.build_ml_features
python -m src.models.train_traditional_models
python -m src.models.train_deep_learning_models --models gru --out resources/experiments/manual_gru
python -m src.analysis.compare_model_performance
python -m src.models.detect_kpi_anomalies
python -m src.analysis.analyze_anomaly_statistics
```

注意：每次 Web 深度学习和异常检测训练都会自动分配唯一的 `resources/experiments/<run_id>/` 输出目录，并保存 `requested_parameters.json`；成功后将可用模型发布到 `resources/models/<run_id>/`。同类历史实验不会被新调参覆盖。命令行复现实验时同样建议指定新的 `--out` 目录。传统 ML 的报告仍沿用固定输出路径，尚未做到完整版本隔离。

## 检查

```bash
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
python -m compileall -q src app.py
```

仓库不包含本地训练数据或权重。代码与文档已统一使用新模块路径，但本次删旧入口后仍需在本机重新执行单元测试和 Web 点击回归。

## 文档

详细操作、模型版本、实时日志和问题排查： [USER_GUIDE.md](USER_GUIDE.md)。
技术结构和迁移准则： [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 第七、八周交付资料

- [项目技术报告](docs/TECHNICAL_REPORT.md)：基于已保存的历史实验数据，标注最新集成尚待验收的内容。
- [收尾验收记录](docs/FINAL_ACCEPTANCE.md)：逐项测试步骤、性能记录、仍需填写的本机验证结果。
- [项目架构](docs/ARCHITECTURE.md)：目录职责和依赖边界。
- 演示 PPT 与 Word 报告另行导出，交付前需随最终测试截图核对。
