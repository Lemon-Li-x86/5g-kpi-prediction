# 5G KPI 预测与异常检测系统使用手册

> 适用分支：`w4/local-integration-2026-10-23`（第四阶段整合版本）。这是**本地离线 KPI 实验与交互系统**，不是实时接入 free5GC/UERANSIM 或运营商网络的线上告警平台。

## 1. 系统包含什么

主要流程：下载数据 → 审计/预处理 → 生成 ML 特征 → 选择算法训练 → 自动保存或归档权重 → 选择本地模型版本 → 上传符合格式的历史 KPI CSV 进行下一小时推理 → 查看和导出结果。

另有历史实验成果展示，包括跨波束平均的逐小时 KPI 曲线、MAE/RMSE/R²、异常候选及相关性分析。**历史结果页面读取的是仓库 `reports/` 中保存的 CSV/JSON，不会自动切换成刚选择的权重的评估指标。**

支持四个 KPI：`DLPRB`、`DLThpVol`、`DLThpTime`、`MR_number`；系统不根据字段名猜测物理单位。

## 2. 文件夹说明

```text
项目根目录/
├── app.py                       Web 主入口
├── src/
│   ├── core/config.py           路径与字段配置
│   ├── data/                    数据下载、审计、预处理和特征
│   ├── models/                  预测、异常检测和推理
│   ├── analysis/                模型比较、异常统计、稳健性验证
│   ├── services/                模型权重、后台任务、流式日志
│   ├── ui/                      界面和历史结果读取
│   └── __init__.py              Python 包标记
├── data/raw/                    原始数据（本地）
├── data/processed/              处理后数据（本地）
├── models/                      历史权重（本地）
├── resources/models/            已归档的模型版本（本地）
├── resources/experiments/       深度学习每次训练的独立实验结果（本地）
├── resources/jobs/              current.json、current.log、stream_session.json 等
├── reports/                     随仓库保存的历史实验结果
└── figures/                     随仓库保存的历史图片
```

`data/`、`models/`、`resources/` 中的本地大文件不会自动上传 GitHub；**重新 Clone 后没有权重是正常的**。不要手动删除或重建现有的 `.venv`、`data/`、`models/`、`resources/`。

## 3. 启动系统

在 VS Code 打开**项目根目录**，在 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

如果已安装过依赖，第一行通常只检查版本和补装缺项。若没有 `.venv`，先安装 Python 3.10+ 并创建或选用你的既有 Python 环境。Web 默认访问 `http://localhost:8501`（具体以终端输出为准）。

实时终端使用浏览器访问本机 `ws://127.0.0.1:<动态端口>`；仅支持本机运行浏览器与服务的这一配置。关闭浏览器不会主动终止已经独立启动的训练进程，但不要在任务执行中删除项目目录。

## 4. 数据管理与预处理

在「数据管理与预处理」检查 `data/raw/` 中四个 gotsf-ds 原始 CSV 是否存在。

- 「下载原始数据」：从配置的数据源下载原始文件。
- 「审计并预处理」：运行检查、清洗、时间对齐、标准化并生成 `data/processed/kpi_timeseries.csv`。
- 「生成传统 ML 特征」：为机器学习预测生成滞后特征（包括 1、2、24 和 168 小时）。

页面当前不支持任意格式 CSV 上传后自动转换为训练集；这是固定公开数据集流水线。

## 5. 模型训练与实时终端

「模型训练」支持：传统 ML、LSTM、GRU、Transformer 和异常检测。点击开始训练后，后台工作器启动固定的受控命令。日志通过 WebSocket 事件流传给浏览器，另外每约 2 秒或达到缓冲阈值写入 `resources/jobs/current.log`。只读终端支持滚动查看，不是可执行任意命令的 Shell。

`resources/jobs/current.json` 记录任务状态；若 WebSocket 服务启动失败，还可检查 `resources/jobs/stream_errors.log`。不要在同一任务尚未结束时反复点击「开始训练」。

**深度学习版本隔离：** 新 LSTM/GRU/Transformer 训练的原始 checkpoint、指标与图表保存于 `resources/experiments/<run_id>/`，成功后将 `.pt` + 同名 `.json` 复制至 `resources/models/<run_id>/` 供推理选择。训练时可以提前停止（Early stopping），不等于训练失败。

传统 ML 和异常检测目前仍写入旧版固定路径 `models/`、`reports/`。开始前会归档现有可信旧模型，完成后将可用权重再归档。固定路径产生的传统实验指标尚未做到完整的逐版本隔离。

## 6. 模型管理与推理

进入「模型版本管理与推理」。下拉框扫描本地 `models/`（legacy）和 `resources/models/`（版本归档）。没有模型时先在本机训练；仓库内已有的历史 CSV 不能还原权重。

选中模型后上传历史 KPI CSV，必须有：

```text
Time,base_station,cell,beam,DLPRB,DLThpVol,DLThpTime,MR_number
```

同一 beam 的时间必须唯一且连续。ML 推理使用多个滞后特征，通常需要至少 **169 个连续小时**历史值；深度学习所需历史长度由相应 `.json` 中的 `seq_len` 与 `seasonal_lag` 决定。结果可导出 CSV。界面上传大小限制为 50 MB。

传统 ML 的 `.pkl` 只能加载本人信任的本机模型文件。DL 的 `.pt` 必须保留同名 JSON 元数据；缺失元数据的文件不会出现在列表。异常模型的完整交互式推理页面目前尚未开放，虽已有 `src.models.inference.detect_anomalies` 接口。

## 7. 历史结果页面

- 「项目总览」：公开数据规模、已有结果和预测异常实验概览。
- 「KPI 预测」：读取已存预测序列，查看 KPI 的逐小时聚合实际/预测曲线。
- 「模型评估」：读取已存指标，比较不同 ML/DL 模型的 MAE、RMSE、R²。
- 「异常检测」：比较 3σ、IQR、Isolation Forest、One-Class SVM 的标记率与候选事件。SVM 来自抽样子集，不能与完整测试集结果直接等量比较。
- 「KPI 关联分析」：相关性和（若文件存在）Holm 校正的统计显著性；相关不代表根因。

这些页面仍是离线历史实验展示，不会自动采用所选权重重新运行对应评估。

## 8. 命令行接口（新模块路径）

```bash
python -m src.data.download_kpi_data
python -m src.data.audit_kpi_data
python -m src.data.preprocess_kpi_data
python -m src.data.build_ml_features
python -m src.models.train_traditional_models
python -m src.models.train_deep_learning_models --models lstm --out resources/experiments/manual_lstm_01
python -m src.models.detect_kpi_anomalies
python -m src.analysis.analyze_anomaly_statistics
```

模块已按职责分层，不再保留 `src/` 根目录的同名 Python 脚本。以前的 `python -m src.<旧脚本名>` 不能再运行，请使用上面的完整模块路径。

直接调用推理 API 示例：

```python
import pandas as pd
from src.models.inference import predict_kpi
history = pd.read_csv("your_history.csv")
result = predict_kpi(history, "resources/models/<run_id>/w3_lstm.pt")
print(result.head())
```

请将 `<run_id>` 替换成真实版本目录名，不能照抄占位符。此 API 不会自动连接实时网络。

## 9. 测试与常见问题

项目根目录执行：

```bash
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
python -m compileall -q src app.py
```

- **找不到模型**：确认 `models/` 或 `resources/models/` 有可识别的权重；DL 要有同名 JSON。
- **LSTM 训练完成却提示 `train_scale_mean` 不同**：旧脚本试图合并标准化协议不同的历史报告；下次通过 Web 启动新的 DL 训练或为命令行指定**新的** `--out` 目录。报错前写出的 `models/w3_lstm.pt` 与 JSON 可以先备份，不要直接覆盖。
- **训练显示 CPU**：当前 PyTorch 未选用 CUDA，不代表 GPU 不可用；需另行核对驱动与 PyTorch CUDA 版本。
- **WebSocket 连接失败**：确认正在本机打开 Web 和浏览器，并查看 `resources/jobs/stream_errors.log`。训练输出通常还在日志中。
- **Web 历史图没有变化**：旧页面读取 `reports/`，不会自动读取独立运行目录的指标。
- **新的结构如何维护**：参阅 `docs/ARCHITECTURE.md`。

## 10. 限制与验收

当前未接入 free5GC/UERANSIM 的实时 KPI 流；异常候选缺少真实故障标签；ML/异常检测完整按版本保存仍需完善；UI 与新模型的指标尚未完全绑定。此分支完成的是**代码整理与功能保留**，并不代表所有真实训练和 Web 点击流程已经在用户电脑上重新验收。
