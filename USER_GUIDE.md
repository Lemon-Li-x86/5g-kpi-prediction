# 5G KPI 预测与异常检测系统使用手册

## 1. 系统定位

本系统是基于开源 KPI 数据集的离线分析与可视化原型。它读取仓库内已生成的预测指标、预测时间序列、异常检测汇总、异常候选事件和相关性分析结果，并通过 Streamlit 提供交互式展示。

当前版本不会在页面打开时训练模型，不会从 free5GC、UERANSIM 或实时基站接口采集数据，也不提供实时告警。异常候选不等同于经过网络故障标签验证的真实故障。

## 2. 目录要求

将文件放在项目根目录，保持以下结构：

```text
5g-kpi-prediction/
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dashboard_data.py
│   ├── preprocess_kpi_data.py
│   ├── train_deep_learning_models.py
│   └── detect_kpi_anomalies.py
├── reports/
│   ├── model_metrics.csv
│   ├── prediction_timeseries.csv
│   ├── w3_deep_learning_metrics.csv
│   ├── w3_gru_hourly_predictions.csv
│   ├── w3_lstm_hourly_predictions.csv
│   ├── w3_transformer_hourly_predictions.csv
│   ├── w3_anomaly_method_summary.csv
│   ├── w3_anomaly_rate_by_hour.csv
│   ├── w3_anomaly_candidate_events.csv
│   ├── w3_anomaly_kpi_correlations.csv
│   └── week3/w3_ml_dl_all_metrics.csv
└── figures/
```

部分结果文件缺失时，对应页面可能显示提示或不显示相关图表。

注意：所有训练与分析脚本从项目根目录使用 `python -m src.<模块名>` 执行；完整流程见 [README.md](README.md)。

## 3. 安装与启动

建议使用 Python 3.10 或更新版本。在项目根目录打开终端，执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Web 应用：

```bash
streamlit run app.py
```

终端会显示本地访问地址。将地址复制到浏览器即可打开系统。若命令找不到 Streamlit，可尝试：

```bash
python -m streamlit run app.py
```

## 4. 功能说明

### 4.1 项目总览

查看数据规模、时间索引范围、训练/测试区间、模型数量和联合异常候选数量，并检查主要结果文件是否存在。

### 4.2 KPI 预测

选择 KPI 和模型，查看测试区间内实际 KPI 均值与预测 KPI 均值的变化曲线，并查看对应的 MAE、RMSE 和 R²。曲线是跨 Beam 的逐小时聚合结果，指标则来自模型在测试样本上的评估；二者统计粒度不同。

横轴是数据集中的整数时间索引，不是实际日历时间。当前数据集采用 Time 0–671 作为训练区间、Time 672–839 作为测试区间。

### 4.3 模型评估

按 KPI 和 MAE、RMSE 或 R² 对比传统机器学习与深度学习模型。MAE 和 RMSE 越低越好，R² 越高越好。页面可下载当前 KPI 的比较表。

### 4.4 异常检测

查看 3σ、IQR、Isolation Forest 和 One-Class SVM 的标记率、逐小时标记率，以及已保存的异常候选事件。One-Class SVM 只在抽样测试子集上评估，不能将其标记率当作与完整测试集同条件的直接比较。

`w3_anomaly_candidate_events.csv` 只保存按 Isolation Forest 分数排序的最多 2,000 条联合候选事件，不代表完整候选集合。筛选后的表格可以导出为 CSV。

### 4.5 KPI 关联分析

查看 KPI Pearson 相关性矩阵。若 `reports/w4_anomaly_kpi_correlation_tests.csv` 已生成，页面还会显示 Holm 校正后的基站聚类稳健显著性结果。若该文件不存在，页面只显示 W3 探索性相关系数，不会虚构 p 值。

相关性不等于因果关系。由于相关性是在异常检测筛选出的候选样本上计算的，结果只能作为探索性分析线索。

## 5. 更新实验结果

在项目根目录运行 `python -m src.detect_kpi_anomalies` 和 `python -m src.analyze_anomaly_statistics` 可以重新产生异常检测与显著性文件。预测训练入口分别为 `python -m src.train_traditional_models` 和 `python -m src.train_deep_learning_models`。完整重训并非启动 Web 所需。

当前 Web 应用读取已保存的 CSV/JSON，不会自动重训模型。若重新运行模型或异常检测脚本，请确认结果已写入 `reports/` 目录，再刷新或重启 Streamlit 页面。

W4 统计诊断脚本完成运行后，会输出相关性显著性结果；将其放在 `reports/` 目录下即可供关联分析页面读取。

## 6. 常见问题

- **找不到结果文件：** 检查文件是否位于项目根目录下的 `reports/`，而不是 `src/reports/`。
- **页面数据没有变化：** 使用 Streamlit 菜单中的重新运行功能，必要时重启应用以清除缓存。
- **某模型没有预测曲线：** 检查对应的 `w3_<model>_hourly_predictions.csv` 或 `prediction_timeseries.csv` 是否存在，以及模型/KPI 名称是否一致。
- **没有相关性显著性：** 这通常意味着 W4 检验文件尚未生成；当前 W3 矩阵本身不包含 p 值。
- **模型数值看起来不像真实物理单位：** 当前图表按结果文件中保存的 KPI 数值尺度展示。正式报告应根据预处理与逆变换过程确认单位，不能仅凭列名推断物理单位。

## 7. 当前限制

1. 仅展示已有离线实验结果，没有在线推理和实时网络采集。
2. free5GC/UERANSIM 尚未连接到当前 KPI 数据处理流程。
3. 仓库中保存的候选事件 CSV 最多包含 2,000 条事件。
4. 数据集没有经过验证的真实故障标签，无法据此报告真实故障检测的 precision、recall 或 F1。
5. 相关性分析是候选样本上的探索性分析，不构成已确认的故障根因结论。

## 第七周部署接口（本地生成模型）

GitHub 历史产物只有指标/预测 CSV 和图片，没有预训练权重。CSV 不能还原模型。
先安装 `pip install -r requirements.txt`，在仓库根目录依次执行：

```shell
python -m src.download_kpi_data
python -m src.preprocess_kpi_data
python -m src.build_ml_features
python -m src.train_traditional_models
python -m src.train_deep_learning_models --out .
python -m src.detect_kpi_anomalies --out .
python -m unittest test_inference -v
```

传统训练会生成 `models/<模型名>_<KPI>.pkl`；深度训练保存最佳权重及同名 JSON 配置，两者必须一起保留。深度标准化现在仅用 validation-start 之前的数据，旧报告不代表修复后重新训练结果。训练命令会更新报告；保留历史结果时请先复制仓库到独立本地工作目录运行。

```python
from src.inference import predict_kpi, detect_anomalies
prediction = predict_kpi(history, 'models/RandomForest_DLPRB.pkl')
# history: DataFrame，包含 Time、base_station、cell、beam 和四个原始 KPI。
# ML 每个波束至少169个连续小时；DL由配置决定窗口和季节历史长度。
# 返回每个波束下一小时原始量纲预测，无需未来标签。
```

异常接口支持与预测相同的原始 KPI DataFrame，自动按已存标准化参数转换并校验波束标识和连续时间；也兼容低层数组接口：`detect_anomalies(values, 'models/anomalies.pkl')`。
values 是与训练完全相同波束顺序、相同标准化的 [波束,时间,4] 数组；首小时为差分上下文、不输出该小时。使用原有三西格玛/IQR（含零IQR回退）、IF、SVM及共识逻辑。统计训练分布保存在本地产物中，推理不重新拟合估计器。数组调用方负责波束顺序和标准化；建议使用 DataFrame 入口以启用元数据校验。

仅加载自己生成且可信的 pickle/PT 产物。models/、权重和pickle均不提交GitHub。
现有 dashboard_data/app.py 继续展示历史报告，尚未调用新接口。

验证边界：仓库没有原始/已处理数据或权重；不能声称已复现历史指标或完成真实数据训练。合成数据测试仅检查模型训练一步后保存/加载的推理一致性。


测试记录（2026-10-22）：Python 3.12；scikit-learn 1.9.1、PyTorch 2.14.1；`python -m unittest test_inference -v`：3 tests，OK（含三个深度网络子案例）。沙箱首次运行因系统临时目录权限失败，获准在沙箱外重跑后通过。`compileall` 通过。未执行真实数据全量训练、XGBoost/LightGBM训练往返、Streamlit交互回归；未完成全仓逐行审查。当前为第一步基础实现，不能视为完整生产部署验收。
