# 文件审计与交付准备（2026-10-25）

## 范围与备份

审计目标为当前项目根目录，包含隐藏文件、.git 和被忽略的运行目录；周报等父目录资料仅按既有报告清单补齐交付文档，不对父目录做清理。原始快照包含 41,660 个文件、3,713,748,556 字节，逐文件 SHA-256 与 tar 内容核对通过后才进行修改。完整库存和工作区补丁在仓库同级 `delivery-safety-20261025/`，交付仓库内保留去除 Git 内部和环境文件的完整项目库存。

## 分类与处理

| 类别 | 处理 |
|---|---|
| 源码、入口、配置、测试 | 保留已有分层及模块路径；保留开始时未提交的算法审查修改 |
| 文档 | README/架构/使用手册保留；从 week4 补齐 10 份修订 Word、10 份 PDF，以及构建所需的 10 份历史原件 |
| 代表性模型、checkpoint、scaler | 全量保留，见 local_artifact_manifest.csv；模型发布方案待确认 |
| 原始及处理数据 | 全部保留；下载、哈希核对与生成步骤见 DATA_AND_MODELS.md |
| 实验指标、日志、图片、CSV | 历史原件不动；从被忽略的 resources 导出 139 份文本/图片到 reports/experiment_archive，原路径仍可用 |
| 重复产物 | SHA-256 确认 68 组，理论重复量 999,462,648 字节；全部保留，待审阅 |
| 可生成缓存 | 69 个 Python 字节码文件、533,629 字节移入仓库外隔离区；无永久删除 |
| 虚拟环境 | 保留且备份，Git 忽略；记录已安装版本 |
| 敏感内容 | 正则扫描工作文本和相关 Git 文本历史；具体边界及发现见 sensitive_scan_initial.json，无法替代完整人工保密审查 |

每个隔离文件的路径、大小、哈希、去向和理由见 `quarantine_manifest.csv`。缓存可能在下次运行重新出现，属于正常行为。大文件未移走；严禁将 duplicate_groups.json 当作自动删除依据，权重同内容也可能被不同 run_id 引用。永久删除隔离区和模型去重均等待用户审阅。

如确需恢复隔离文件，在仓库根目录运行以下命令。脚本先检查整份清单、路径和哈希；若同路径缓存已经重新生成，会拒绝覆盖，保留当前文件与隔离副本。

```powershell
.\.venv\Scripts\python.exe scripts/audit_delivery.py --restore-quarantine reports/delivery_audit_20261025/quarantine_manifest.csv --quarantine-root ../delivery-safety-20261025/quarantine
```

后续只读盘点可运行 `python scripts/audit_delivery.py --output ../delivery-inventory-new.json`，该命令不清理文件，且拒绝覆盖已有清单。

## 引用、参数和模块

入口仍为 `app.py`，调用 data → models → analysis，services 管理训练及发布，ui 展示。`import_inventory.json` 记录导入关系，`path_reference_inventory.json` 记录静态字面路径，动态拼接路径不能凭此判为无引用。现有数据/模型注册/报告读取路径保持可用。

将 DL、异常检测与传统 ML 的 20 项 CLI 默认值改为读取现有 `core/training_config.py`，替换前逐项断言数值相同；未改变历史模型参数、结构、切分或训练算法。参数更改仍写入各 run 的 config 与权重侧车。固定研究协议/既有实验记录中的参数继续独立保存，以免未来默认值变化改写历史含义。现有模型函数内部默认值仍保留，未为统一形式重写算法。

修正 `.gitignore` 的根目录限定，避免 `models/` 误屏蔽新源码；继续忽略数据、权重、缓存、密钥和虚拟环境。实验可交付副本显式存放在 reports 下。补齐文档/调参可选依赖；WebSocket 上限与当前已核验的 17.3 版本对齐。Word 构建器改用仓库自带的 docs/source_reports，支持 --source-root，不再依赖本机固定盘符。

## 文档一致性

修订 Word/PDF 逐份按既有清单核对。共有 1 份当前文件与历史清单哈希不同，明细见 recovered_documents.json。历史清单不重写；当前完整技术报告 Word 原样保留，PDF 与原记录匹配。本次没有编辑或重新渲染这些文档，二者是否逐字一致需人工复核。

## Git 交付边界

起点为 w4/local-integration-2026-10-23 的 0781ca9，开始已有未提交修改；现有 main 的后续改动仅为技术报告格式和文字修订。远程为 https://github.com/Lemon-Li-x86/5g-kpi-prediction.git，GitHub 接口核实为 private、连接权限含 push/admin。交付使用 codex/ 前缀的新本地分支；本轮以本地 commit 为完成目标，未推送，未改变远程可见性。

提交包括必要代码、既有未提交审查及测试、所有仓库内文档、报告/图表、补齐的阶段交付物、实验文本/图像镜像和审计清单；不含虚拟环境、原始大数据、二进制权重、完整备份、隔离缓存。提交前扫描、文件大小和验证结论见同目录 validation 与提交范围记录。推送前必须展示分支、提交范围，并确认目标仓库和私有/公开选择。

## 审阅入口

详细清单位于 `reports/delivery_audit_20261025/`：file_inventory_before.csv、duplicate_groups.json、large_files_retained.json、local_artifact_manifest.csv、quarantine_manifest.csv、exported_evidence.json。副本为交付证据，不用于 Web 运行时自动发现。若继续实验，新证据须再次归档，避免仅保留在被忽略的 resources。


## 本轮验证结论

58 项自动测试（tests 中 53 项，加根目录推理/注册 5 项）、三类训练 CLI 帮助入口、pip check、10 项实际权重推理与 CSV 往返、六个页面 AppTest、W4 已保存统计一致性均通过。640 份原始数据、模型和实验证据再次核对哈希，没有改变；所有候选提交文件小于 100 MiB。新代码语法及忽略规则验证、敏感模式扫描通过，详见 validation.json。

首次测试受 Windows 临时目录权限限制，改用仓库外可写临时目录后通过；界面 AppTest 在受限执行环境停滞并被中断，在受限环境外进行本机验证后六页通过。日志均保留；没有重新训练，也没有进行浏览器真人操作验收。已有 Streamlit 弃用提示记录于日志，本次没有扩大到 UI API 迁移。

逐文件重复产物审阅表为 cleanup_review_candidates.csv，仅说明内容重复；各 run_id 仍可能被文档、模型注册或动态路径使用，不能直接批量删除。缓存的实际隔离记录独立见 quarantine_manifest.csv。

新增证据目录通过 .gitattributes 保留原始字节，防止 Windows 自动换行转换导致副本哈希失效。原有已跟踪历史报告继续沿用已有换行策略，本机原文件哈希与跨平台规范化 Git blob 哈希不可混为一谈。
