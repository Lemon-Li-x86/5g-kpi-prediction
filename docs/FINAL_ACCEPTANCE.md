# 第七、八周系统收尾与验收记录

> 状态：待本地实际验收。此表明确区分**仓库已有代码**、**历史实验数据**与**今天在 Windows 上重新验证的结果**。勿将待测试项目标记为通过。

## 一、对应任务书交付要求

| 原周次 | 任务书要求 | 仓库现状 | 收尾验收 |
|---|---|---|---|
| W7 | 预测与异常检测模块集成 | 预测和异常检测各有训练脚本与推理 API；Streamlit 已加入对应交互入口 | [ ] 实际输入 CSV 分别调用两种推理并成功导出 |
| W7 | 可运行的 Streamlit Web | `app.py`、`src/ui/`、`src/services/` 已存在 | [ ] Windows 本机启动、八个菜单均正常 |
| W7 | 系统测试与性能优化 | 已有单元测试、UTF-8 兼容、WebSocket 日志、版本化输出 | [ ] 全套测试通过，记录耗时、峰值内存、设备及异常处理 |
| W7 | 系统使用手册 | `USER_GUIDE.md` 已编写 | [ ] 跟随手册完成实际操作，修复错误步骤 |
| W8 | 完整技术报告 | `docs/TECHNICAL_REPORT.md` | [ ] 对照实际测试填写最终状态 |
| W8 | 项目演示 PPT | 本轮按用户要求暂缓，先完成代码、技术报告和使用手册 | [ ] 后续另行处理 |
| W8 | GitHub 代码和文档 | `w4/local-integration-2026-10-23` | [ ] `git status` 干净、版本固定、README 链接齐全 |

## 二、最小验收命令（项目根目录，激活原虚拟环境）

```powershell
git status --short
python -m compileall -q src app.py
python -m unittest discover -s tests -v
python -m unittest test_inference test_model_registry -v
python -m streamlit run app.py
```

以上命令执行结果应保存到 `resources/jobs/` 或单独截图。**不要在已有训练运行时替换它的源文件**。由于  `resources/` 被 Git 忽略，测试证据不会自动上传 GitHub。

检查依赖：`python -m pip check`。

## 三、准备 Web 验收 CSV

```powershell
python -m src.data.prepare_demo_inputs
```

分别使用生成的 `resources/demo_inputs/forecast_history.csv` 和 `resources/demo_inputs/anomaly_history.csv` 验证两类推理。异常 CSV 包含训练数据全部波束在最后三个完整小时的 KPI，首小时作为差分上下文。该脚本不会重训模型或生成故障标签。

## 四、Web 手动操作验收

- [ ] 「数据管理与预处理」能识别本地四个原始 KPI 文件与已有预处理 CSV。不要为检验 UI 不必要地重复下载大型数据。
- [ ] 「模型训练」可选择 LSTM/GRU/Transformer，实际提交 `epochs=3`、`batch_size=64`、`patience=2` 等测试参数。日志打印出的命令、`current.json` 与 `requested_parameters.json` 一致。
- [ ] 模型训练日志通过本机 WebSocket 实时更新、允许向上滚动，结束时显示成功或失败。
- [ ] 新训练不覆盖旧 checkpoint，产生唯一 `resources/experiments/<run_id>/` 与 `resources/models/<run_id>/`，包含权重及同名元数据。
- [ ] 「模型管理与推理」能枚举本地版本化的 .pt、.pkl，选择一份受信任权重，使用合适长度的 KPI 历史 CSV 输出预测并下载。
- [ ] 「模型管理与推理」选择 `anomalies.pkl`，上传与模型训练 **完全相同的波束集合**、小时轴对齐、至少包含一个上下文小时的 CSV，输出 3σ / IQR / IF / OCSVM / consensus 标记并导出。候选不是确认故障。
- [ ] 五个历史结果页面（项目总览、KPI 预测、模型评估、异常检测、KPI 关联分析）可正确读取原有 reports、figures。
- [ ] 提交非法超参数时阻止训练，显示可理解报错；重复点击开始训练不会创建多个作业。
- [ ] 断开浏览器后后台训练仍可正常结束；重新打开能查到最后状态与已保存日志。
- [ ] 上传缺少 KPI 字段或时间不连续的数据时显示可诊断错误。

## 五、性能记录模板（需本机填写，勿杜撰）

| 操作 | 样本数 | CPU / GPU | 耗时（s） | 峰值 RSS / GPU 显存 | 成功 / 失败 | 证据路径 |
|---|---:|---|---:|---|---|---|
| 原始数据预处理 | 待填 | 待填 | 待填 | 待填 | 待测 | |
| 传统 ML 快速模式或既有任务 | 待填 | 待填 | 待填 | 待填 | 待测 | |
| GRU 3 Epoch 调试训练 | 待填 | 待填 | 待填 | 待填 | 待测 | |
| 模型加载及 1 次下一小时推理 | 待填 | 待填 | 待填 | 待填 | 待测 | |
| 异常检测 1 个批次 | 待填 | 待填 | 待填 | 待填 | 待测 | |
| Web 首次加载及页面切换 | 待填 | 待填 | 待填 | 待填 | 待测 | |

设备检查：

```powershell
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

不要声称使用了 RTX 4060 GPU，除非输出实际显示 `cuda: True` 且训练日志确认模型使用 CUDA。

## 六、当前已知缺口与风险

1. 原始仿真 free5GC/UERANSIM 连通性实验与此 gotsf-ds 公开数据预测流水线**并非实时直接连接**；别把离线分析称作实时核心网告警平台。
2. 异常候选没有真实故障标签，不能报告 Precision、Recall 或 F1，也不能把 KPI 相关性称为确定根因。
3. 历史评价 CSV 不会自动随模型下拉框切换；展示时应明确「历史实验」。
4. 传统 ML 仍输出到部分固定 `reports/` 路径，可能与版本化深度学习实验不同；归档与报告应分别核对。
5. 新增的 Web 异常推理入口和训练参数面板还需本机实测，不因代码已合并就默认验收通过。
6. Python 3.14/Windows 下的第三方库兼容性、UTF-8 控制台输出及 GPU 环境必须由本机确认。

## 七、完成判定与交付封版

通过最低验收链路后，建议固定一个 Git 标签 `w4-final-2026-10-23`，再将 **经过核实的**测试截图、日志摘要、报告、PPT 添加到交付目录。未经复测的项应写明「未测试 / 仅单元测试」而非打钩。最终将这份记录连同报告一起发送导师。
