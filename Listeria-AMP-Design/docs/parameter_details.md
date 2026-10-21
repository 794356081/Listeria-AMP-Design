# 参数与输出字段说明

本文档列出 `config.yaml` 的全部参数、各脚本的命令行覆盖项、阶段输出文件及其字段含义。默认值与 `config.yaml` 一致。

---

## 1. 全局配置

| 配置键 | 默认值 | 说明 |
|---|---|---|
| `project.seed` | `20261019` | Python / NumPy / PyTorch 随机种子 |
| `paths.protgpt2_model` | `nferruz/ProtGPT2` | Hugging Face 模型 ID 或本地目录 |
| `paths.esm2_model` | `facebook/esm2_t33_650M_UR50D` | ESM-2 模型 ID 或本地目录 |

模型也可以分别用 `01_sequence_generation.py --model` 与 `03_esm2_embedding.py --model` 覆盖。离线环境请配合 `--local-files-only` 使用本地权重目录。

---

## 2. 生成阶段（`generation`）

| 配置键 | 默认值 | 说明 |
|---|---:|---|
| `target_sequences` | 100000 | 目标唯一序列数 |
| `min_length` | 10 | 最短长度（aa） |
| `max_length` | 50 | 最长长度（aa）；超出截断 |
| `prefix_length` | 3 | 提示前缀长度（aa） |
| `prompts_per_batch` | 20 | 每批提示数 |
| `samples_per_prompt` | 10 | 每个提示的采样数 |
| `temperature` | 0.85 | 采样温度 |
| `top_p` | 0.9 | 核采样概率 |
| `top_k` | 40 | 每步保留的最高概率词元数 |
| `repetition_penalty` | 1.2 | 重复惩罚 |
| `max_new_token_choices` | `[12, 15, 18, 20]` | 每批随机选择的生成长度 |
| `checkpoint_every_batches` | 10 | 进度 JSON 写入间隔 |

命令行覆盖：

| 参数 | 说明 |
|---|---|
| `--config` | 配置文件路径 |
| `--seeds` | 参考肽 FASTA 或 CSV（CSV 需含 `sequence` 列） |
| `--out-dir` | 输出目录 |
| `--model` | ProtGPT2 模型 ID 或本地目录 |
| `--target` | 覆盖目标序列数 |
| `--max-batches` | 批次上限（0 表示不限制），用于小规模试跑 |
| `--device {auto,cuda,cpu}` | 推理设备 |
| `--seed` | 覆盖随机种子 |
| `--local-files-only` | 只从本地加载模型 |

**输出**

| 文件 | 内容 |
|---|---|
| `generated_sequences.fasta` | 唯一序列；头部含 `len`、`seed`、`batch` |
| `generation_progress.json` | 目标、已完成数、批次数、耗时 |
| `generation_stats.json` | 长度分布、去重率、参数快照 |
| `generation.log` | 运行日志 |

---

## 3. 理化初筛（`physchem`）

| 配置键 | 默认值 | 说明 |
|---|---:|---|
| `min_length` | 10 | 最短长度 |
| `max_length` | 50 | 最长长度 |
| `charge_min` | −2.0 | 净电荷下限（pH 7.4） |
| `charge_max` | 6.0 | 净电荷上限 |
| `gravy_min` | −1.5 | GRAVY 下限 |
| `gravy_max` | 1.0 | GRAVY 上限 |
| `max_homopolymer_run` | 5 | 允许的最大连续相同残基数（达到该值即剔除） |

命令行参数：`--input`、`--out-dir`、`--config`、`--exclude`（参考肽 FASTA/CSV，用于剔除完全相同的序列）。

**输出**

| 文件 | 字段 |
|---|---|
| `screened_physchem.fasta` | 通过序列，头部为原标识 |
| `screened_physchem.csv` | `id, sequence, length, MW, pI, net_charge, GRAVY, muH, helix_prop, cationic_ratio, hydrophobic_ratio` |
| `filter_summary.json` | 输入数、通过数、通过率、各规则触发次数 |
| `filter.log` | 运行日志 |

`MW` 为残基质量求和加一分子水；`pI` 为净电荷为零的 pH；`muH` 为 11 残基窗、100° 的疏水矩；`helix_prop` 为 Chou–Fasman 螺旋倾向均值。

---

## 4. 表征阶段（`embedding`）

| 配置键 | 默认值 | 说明 |
|---|---:|---|
| `model` | `facebook/esm2_t33_650M_UR50D` | ESM-2 模型 |
| `batch_size` | 100 | 初始批大小；OOM 时自动减半 |
| `max_length` | 1024 | 分词最大长度 |
| `pooling` | `mean_excluding_special_tokens` | 池化方式（当前仅实现该方式） |

命令行覆盖：`--input`、`--out-dir`、`--config`、`--model`、`--batch-size`、`--max-length`、`--device`、`--local-files-only`、`--overwrite`。

**输出**

| 文件 | 内容 |
|---|---|
| `esm2_embeddings.npy` | `float32`，形状 `(N, 1280)` |
| `esm2_embedding_index.csv` | `row_index, peptide_id, length, sequence` |
| `embedding_progress.json` | 完成数、维度、耗时、NaN 行数 |
| `embedding.log` | 运行日志 |

---

## 5. 评分阶段（`scoring`）

| 配置键 | 默认值 | 说明 |
|---|---:|---|
| `weights.esm_similarity` | 0.45 | W1 权重 |
| `weights.cationic_amphipathic` | 0.20 | W2 权重 |
| `weights.helix_propensity` | 0.15 | W3 权重 |
| `weights.hydrophobicity_balance` | 0.10 | W4 权重 |
| `weights.low_hemolysis` | 0.10 | W5 权重 |
| `high_activity_quantile` | 0.5 | 高活性参考集的 MIC 分位数 |
| `hydrophobicity_balance.target_gravy` | 0.2 | W4 高斯中心 |
| `hydrophobicity_balance.sigma` | 0.5 | W4 高斯宽度 |
| `hemolysis_model.min_labels` | 16 | 训练溶血模型所需最少标签数 |
| `hemolysis_model.min_cv_auc` | 0.60 | 启用模型的最低交叉验证 AUC |
| `hemolysis_model.C` | 0.1 | 逻辑回归正则强度 |
| `hemolysis_model.max_iter` | 3000 | 逻辑回归最大迭代 |
| `hemolysis_model.n_splits` | 5 | 交叉验证折数 |
| `top_n` | `[100, 500]` | 预留的分级规模（当前脚本用 `--top-n`） |

命令行参数：`--physchem`、`--embeddings`、`--embedding-index`、`--known-amps`、`--known-embeddings`、`--known-embedding-index`、`--out-dir`、`--config`、`--top-n`。

**参考肽表要求**

| 列 | 必填 | 说明 |
|---|---|---|
| `peptide_id` | 是 | 唯一标识，需与参考表征索引一致 |
| `sequence` | 是 | 标准单字母序列 |
| `mic_ug_ml` | 是（W1 需要） | 抗李斯特菌 MIC，μg/mL |
| `hemolytic` | 否 | 1/0；缺失则不计入溶血模型 |
| `source` | 否 | 来源数据库或文献 |
| `length` | 否 | 序列长度 |

**输出**

| 文件 | 内容 |
|---|---|
| `scoring_result.csv` | 全部候选，按 `score` 降序 |
| `top_candidates.csv` | 前 `--top-n` 条 |
| `scoring_summary.json` | 候选数、参考集规模、MIC 中位数、高活性阈值、溶血模型方法、CV AUC、分值范围、权重 |
| `scoring.log` | 运行日志 |

**`scoring_result.csv` 字段**

| 字段 | 含义 |
|---|---|
| `rank` | 排名（1 为最高分） |
| `peptide_id` | 候选标识 |
| `sequence` | 序列 |
| `length` | 长度（aa） |
| `MW` | 分子量（Da） |
| `pI` | 等电点 |
| `net_charge` | pH 7.4 净电荷 |
| `GRAVY` | Kyte–Doolittle 平均疏水性 |
| `muH` | 疏水矩 |
| `helix_prop` | Chou–Fasman 螺旋倾向均值 |
| `cationic_ratio` | (K+R)/长度 |
| `hemolysis_prob` | 溶血概率（模型或启发式） |
| `W1_esm_similarity` | ESM-2 相似度百分位秩 |
| `W2_cationic_amphipathic` | 阳离子两亲性百分位秩均值 |
| `W3_helix_propensity` | 螺旋倾向百分位秩 |
| `W4_hydrophobicity_balance` | 疏水平衡高斯得分 |
| `W5_low_hemolysis` | 1 − 溶血概率 |
| `score` | 综合得分（0–100） |

---

## 6. 运行规模与资源

| 阶段 | 参考运行规模 | 资源备注 |
|---|---|---|
| 生成 | 100,021 条唯一短肽 | ProtGPT2 FP16，单批 200 条 |
| 初筛 | 100,021 → 54,181 | CPU，秒级至分钟级 |
| 表征 | 54,181 × 1280 | ESM-2 650M FP32；支持断点续跑 |
| 评分 | 54,181 条 | CPU，秒级至分钟级 |

演示数据规模为 40 条输入、30 条候选、29 条参考肽，第 2 与第 4 步可在无 GPU 环境下数秒完成。

---

## 7. 路径与可移植性

- 所有脚本默认从仓库根目录运行；`--config config.yaml` 使用相对路径。
- 输出目录、模型目录、输入文件均可通过命令行指定，脚本内部不含绝对路径。
- 若在 Windows 上运行，请使用 PowerShell/CMD 并将示例命令中的行继续符 `\` 改为 `` ` ``，或写成单行。
