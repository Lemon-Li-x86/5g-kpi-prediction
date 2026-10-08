# 压缩四周项目 · 第二周实验报告

**项目名称：** 基于 AI 的 5G 网络关键性能指标预测与异常检测系统  
**汇报周期：** 压缩项目第2周（覆盖原8周方案中的第3–4周）  
**整理日期：** 2026-10-08（北京时间）  
**阶段：** KPI 数据集获取与预处理、传统机器学习 KPI 预测

## 一、本周工作目标

按照原始项目任务书，第3周需完成公开数据获取、清洗、归一化和时间序列数据集构建；第4周需完成传统 ML 预测、MAE/RMSE/R² 评估和预测可视化。本周将上述两部分合并交付。

## 二、完成的工作

### 1. KPI 数据集审计与预处理

选取公开 **gotsf-ds** 数据集的 `DLPRB`、`DLThpVol`、`DLThpTime`、`MR_number` 四种 KPI。原始输入为四份含840个时间索引和2880个 beam 的宽表 CSV；beam ID 解析为基站、cell 和 beam 三个层级。

已实现 `src/audit_raw_data.py` 与 `src/preprocess.py`，包括：
- 检查缺失、无穷、负值、重复时间、连续性与 beam ID 格式；
- 保留审计通过的零值及长尾高值，不擅自视为无效异常；
- 将宽表整理为 `Time × beam` 的长表，合计 **2,419,200行、17列**；
- 按 `Time=0–671` / `672–839` 做时间顺序划分；
- 用训练段拟合 StandardScaler 和 MinMaxScaler，生成标准化、归一化两种特征；
- 生成 `reports/raw_data_audit.json`、`reports/preprocessing_summary.json` 和 `reports/week3_data_analysis.md`。

原有审计结果记录未发现缺失值、无限值、负数、重复时间与时间缺口。**上述为仓库中已有实验产物的记录，本次周报整理没有重新运行全量数据审计。**

### 2. 传统 ML 特征构造和预测

`src/build_ml_features.py` 针对每个 beam 的四个 KPI 构造五个历史值：`t,t-1,t-2,t-24,t-168`，共20个特征，以 `t+1` 为回归目标。最大滞后及预测目标决定有效样本为1,932,480行；训练样本1,448,640，测试样本483,840。

`src/train_ml_models.py` 实现 Persistence、LinearRegression、DecisionTree、RandomForest、XGBoost 和 LightGBM 六种比较方法，4个 KPI 共24组记录。实验配置标注为全量模式（`quick_mode=false`）。

### 3. 评估与结果图

现有 `reports/model_metrics.csv` 包括 MAE、RMSE、R²，以及拟合和预测耗时；仓库 `figures/` 内有四类 KPI 的预测曲线和三个误差比较图。

本周进一步形成 `reports/week4_model_evaluation.md`，集中解释实验设置、模型优劣、数据切分与评估限制。

## 三、核心实验结果

| 预测 KPI | MAE 最优方法 | 最优 MAE | 相对 Persistence 改善 | RMSE 最优方法 | 最优 RMSE | 对应 R² |
|---|---|---:|---:|---|---:|---:|
| DLPRB | XGBoost | 0.261817 | 12.21% | RandomForest | 0.493705 | 0.787133 |
| DLThpVol | RandomForest | 0.207615 | 12.68% | RandomForest | 0.460162 | 0.615323 |
| DLThpTime | XGBoost | 0.243576 | 12.78% | RandomForest | 0.524467 | 0.702391 |
| MR_number | XGBoost | 0.224767 | 8.58% | RandomForest | 0.395647 | 0.867182 |

四类 KPI 的最佳 MAE 均低于持久性基线；随机森林在全部四项指标上取得最低 RMSE 及最高 R²。最优方法依赖选用指标，不能简单将某一个模型表述为全面最优。

注意：上表的“对应 R²”属于 **RMSE 最优模型（RandomForest）**，而不一定属于 MAE 最优模型。统计指标基于 beam-level 测试行计算；时序图则对每个测试时刻的所有 beam 求均值，二者粒度不同。

## 四、工程检查和问题

1. **数据量与版本控制**：本地已保存原始和预处理 CSV，最终通过交付压缩包提供；Git 仓库用 `.gitignore` 排除大体积数据文件，不构成交付缺失，但需要在压缩包中核对文件路径。
2. **避免数据泄漏**：预处理只在训练期拟合 scaler；预测特征限制在目标时刻之前，并以目标时刻定义训练/测试切分。
3. **数据边界**：测试集为公开数据集训练文件内部的最后一周留出段，并非独立官方测试集。
4. **模型评估边界**：目前未建立滚动回测和单独验证集，后续不能反复利用同一测试集做参数调优并声称具有无偏泛化结果。
5. **尚待本机核验**：依赖版本、完整数据包可再运行性及各交付文件的最终打包一致性需要提交前进行本机检查。

## 五、阶段性结论

根据仓库已有运行记录，原项目第3周的数据预处理与第4周的传统机器学习预测已形成较完整的可审计成果，已能为后续 LSTM/GRU 和异常检测提供数据与经典机器学习对照。接下来优先完成依赖及复现文档、交付包自检，而非无必要地重复运行相同实验。

## 六、下一周计划（原第5–6周）

- 用相同的预测目标、时间切分、测试样本和评估指标开发 LSTM/GRU 模型；如计算资源允许，尝试 Transformer 并比较传统 ML 基线。
- 构建3σ/IQR等统计检测方案及 Isolation Forest / One-Class SVM 机器学习异常检测，明确异常定义和检出率指标。
- 初步用 KPI 关联关系形成根因候选解释，避免将相关性描述为已经证明的真实故障因果。
- 整理下一次汇报需要的结果表、图和周报。

## 七、交付物索引

| 原方案周次 | 必需交付物 | 本次对应路径 |
|---|---|---|
| 第3周 | 数据预处理 Python 脚本 | `src/audit_raw_data.py`; `src/preprocess.py` |
| 第3周 | 预处理 KPI CSV | 本地 `data/processed/kpi_timeseries.csv`（压缩包中核对） |
| 第3周 | 数据集统计分析报告 | `reports/week3_data_analysis.md` |
| 第4周 | 传统 ML 预测模型代码 | `src/build_ml_features.py`; `src/train_ml_models.py` |
| 第4周 | 模型性能评估报告 | `reports/week4_model_evaluation.md`、`reports/model_metrics.csv` |
| 第4周 | 预测可视化 | `figures/*_predictions.png` 与 `figures/model_comparison_*.png` |

**注：** 该报告的“完成”指仓库内已有对应源代码与实验产物，当前在线整理不等于在用户电脑上重新执行或全面复核。
