# 第三阶段：传统机器学习与深度学习 KPI 预测对比分析

## 数据来源

- 传统机器学习：`D:\1_ToGo\Huawei5G\py\reports\model_metrics.csv`
- 深度学习：`D:\1_ToGo\Huawei5G\py\reports\w3_deep_learning_metrics.csv`
- 评价粒度：每个 beam × 目标小时，使用原始 KPI 尺度。

## 三项指标的最优模型比较

| KPI | 指标 | 传统 ML 最优 | DL 最优 | DL 相对变化 |
|---|---|---|---|---:|
| DLPRB | MAE | XGBoost (0.261817) | GRU (0.245618) | +6.19% |
| DLPRB | RMSE | RandomForest (0.493705) | GRU (0.468861) | +5.03% |
| DLPRB | R2 | RandomForest (0.787133) | GRU (0.808018) | +0.0209 |
| DLThpVol | MAE | RandomForest (0.207615) | GRU (0.204574) | +1.46% |
| DLThpVol | RMSE | RandomForest (0.460162) | LSTM (0.484384) | -5.26% |
| DLThpVol | R2 | RandomForest (0.615323) | LSTM (0.573760) | -0.0416 |
| DLThpTime | MAE | XGBoost (0.243576) | GRU (0.229781) | +5.66% |
| DLThpTime | RMSE | RandomForest (0.524467) | GRU (0.513352) | +2.12% |
| DLThpTime | R2 | RandomForest (0.702391) | GRU (0.714872) | +0.0125 |
| MR_number | MAE | XGBoost (0.224767) | GRU (0.209693) | +6.71% |
| MR_number | RMSE | RandomForest (0.395647) | GRU (0.366888) | +7.27% |
| MR_number | R2 | RandomForest (0.867182) | GRU (0.885789) | +0.0186 |

## 实验解释及局限性

- Checked metadata: identical held-out target hours 672..839, 483,840 beam-hour observations.
- DL training windows: 60000; ML training rows: 1448640.
- MAE/RMSE: lower is better; R2: higher is better.
- Comparison describes held-out accuracy, NOT equal-data/equal-compute architecture superiority: DL and ML have different training-sample counts, input representations and model-selection procedures.
- Data were chronologically held out from files named 'train'; the evaluation is not an independent official test set.
- Original DL StandardScaler was fitted on hours 0..671, including the internal DL validation period. This limitation should be disclosed.

## 下一步分析

- 分析 DLThpVol 预测的误差分布，尤其是 MAE 和 RMSE 不同步的现象。
- 在仅由训练时间段拟合标准化器的条件下重复 DL 实验。
- 若要论证模型稳定性，增加多个时间窗口的滚动验证或重复种子试验。
- 不把模型预测偏差直接当作真实 5G 网络故障标签。
