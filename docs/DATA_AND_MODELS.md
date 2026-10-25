# 数据、模型与复现

代码和文档随 Git 仓库交付；原始 CSV、处理数据、模型/scaler 和逐样本配对预测保留在本机，未上传。完整校验清单见 `reports/delivery_audit_20261025/local_artifact_manifest.csv`。清单包括相对路径、大小、SHA-256；不要仅凭相同文件名替换权重。

## 新机器准备

在仓库根目录使用 Python 3.10+ 创建虚拟环境，再安装 `requirements.txt`。本机核验环境为 Windows、Python 3.14、CPU，完整版本快照见 `reports/delivery_audit_20261025/environment_freeze.txt`，它是环境记录，不保证跨平台直接安装。PyTorch 可按目标平台准备相容构建。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.data.download_kpi_data
.\.venv\Scripts\python.exe -m src.data.audit_kpi_data
.\.venv\Scripts\python.exe -m src.data.preprocess_kpi_data
.\.venv\Scripts\python.exe -m src.data.build_ml_features
```

下载源由 `src/core/config.py` 的 `DATASET_REPO_ID=netop/gotsf-ds`、`data/train` 和四文件清单统一定义，KPI 为 DLPRB、DLThpVol、DLThpTime、MR_number。默认下载镜像由脚本设置为 hf-mirror.com；可预先设置 `HF_ENDPOINT=https://huggingface.co`。本次未核实原始下载的远端 revision；须用本机原始 CSV 哈希核对数据版本，哈希不符时不能宣称完全复现历史实验。数据再分发须另核对上游许可。

训练时为每次实验指定新的 `--out resources/experiments/<unique_run_id>`；完整命令见 README。研究复核入口为 `python -m src.analysis.final_model_validation`，会执行较长训练并写入既定审查目录，运行前应保存原结果。调参及文档工具的额外依赖位于 `requirements-research.txt`；本机主环境未安装这些可选包，本次未重跑调参或重建 Word/PDF。

## 权重和证据的对应关系

- 历史模型：`models/`；已发布模型：`resources/models/<run_id>/`；Web 运行原件：`resources/experiments/<run_id>/`。
- DL `.pt` 必须携带同名 `.json`，包括结构参数、训练 mean/scale；ML、异常检测 `.pkl` 应使用原 bundle，不能换用新 scaler。仅加载可信来源的模型。
- 研究复核每折/种子的 `resources/final_audit_20261025/` 权重、`beam_keys.csv`、`paired_predictions.npz` 必须成套保留。Persistence 是无训练基线，不能因没有 checkpoint 判定缺失。
- 传统 ML 五类模型、LSTM/GRU/Transformer、3σ/IQR/IF/OCSVM 的历史对比仍在 `reports/`、`figures/`；新研究不覆盖旧成绩。新运行的文本和图片证据镜像在 `reports/experiment_archive/`，含原始路径和哈希的索引见 `exported_evidence.json`。

## 大文件交付策略

本次未删除任何数据或权重，也未配置 LFS、迁移 Git 历史或上传二进制。模型和逐样本预测总量较大，建议在用户确认后将一套代表性完整 bundle 放到私有 Release 或外部存储，再把下载位置、版本与 SHA-256 写回清单；若选择 LFS，再针对选定文件配置规则。当前本机备份不能充当其他人可下载的模型地址，因此新克隆可从代码训练，但不能直接使用这些本机历史权重。

完整原始快照位于仓库同级 `delivery-safety-20261025/full-project-before-cleanup.tar`，包含 .git、虚拟环境、忽略文件及全部运行产物，校验文件见同级 `BACKUP_VERIFIED.json`。恢复时解包到新空目录并核对 `verified_snapshot_manifest.json`，不要覆盖当前工作目录。虚拟环境包含绝对路径，跨机器应重新创建。
