# 示例结果说明

本目录提供一组小规模、可离线复现的演示结果，用于检查管线各阶段的文件格式与评分逻辑，不代表完整库的筛选结论。

## 1. 文件清单

| 文件 | 说明 |
|---|---|
| `example_scoring_result.csv` | 30 条候选的五指标评分与排序结果 |
| `result_description.md` | 本说明 |
| `example_candidate_embeddings.npy` | 30 × 1280 的候选 ESM-2 表征（`float32`） |
| `example_candidate_embedding_index.csv` | 候选表征的行索引、标识、长度与序列 |
| `example_known_embeddings.npy` | 29 × 1280 的参考肽 ESM-2 表征（`float32`） |
| `example_known_embedding_index.csv` | 参考表征的行索引、标识、长度与序列 |

预计算表征由 ESM-2 650M 生成，目的是让第 2、4 步可以在没有 GPU 与模型权重的情况下运行。若需要从头复现第 3 步，请按 README 的完整流程对 `screened_physchem.fasta` 与 `known_amps_example.fasta` 重新表征。

## 2. 示例数据来源

- `data/example_input.fasta`：40 条演示输入，包含 30 条可通过理化初筛的序列与 10 条会被剔除的序列（电荷、GRAVY 或连续重复不满足阈值）。
- `data/known_amps_example.csv`：29 条已知抗李斯特菌肽示例子集，含 MIC 与溶血标签，来源于 DRAMP、APD 与 DBAASP 公共数据库。
- 示例候选与参考肽均来自本项目的参考运行结果，仅抽取小规模子集用于演示。

## 3. 复现命令

```bash
# 第 2 步：理化初筛（40 → 30）
python src/02_physicochemical_filter.py \
  --input data/example_input.fasta \
  --exclude data/known_amps_example.csv \
  --out-dir runs/demo/02_filter

# 第 4 步：使用预计算表征评分
python src/04_scoring_ranking.py \
  --physchem runs/demo/02_filter/screened_physchem.csv \
  --embeddings examples/example_candidate_embeddings.npy \
  --embedding-index examples/example_candidate_embedding_index.csv \
  --known-amps data/known_amps_example.csv \
  --known-embeddings examples/example_known_embeddings.npy \
  --known-embedding-index examples/example_known_embedding_index.csv \
  --out-dir runs/demo/04_scoring
```

`runs/demo/04_scoring/scoring_result.csv` 应与 `example_scoring_result.csv` 的排序和分值一致。浮点末位可能因平台与库版本略有差异。

## 4. 字段说明

| 字段 | 含义 |
|---|---|
| `rank` | 排名，1 为最高分 |
| `peptide_id` | 候选标识 |
| `sequence` | 氨基酸序列 |
| `length` | 长度（aa） |
| `MW` | 分子量（Da） |
| `pI` | 等电点 |
| `net_charge` | pH 7.4 净电荷 |
| `GRAVY` | Kyte–Doolittle 平均疏水性 |
| `muH` | 疏水矩（11 残基窗，100°） |
| `helix_prop` | Chou–Fasman 螺旋倾向均值 |
| `cationic_ratio` | (K+R)/长度 |
| `hemolysis_prob` | 溶血概率（逻辑回归或理化启发式） |
| `W1_esm_similarity` | 中心化 ESM-2 余弦相似度的库内百分位秩 |
| `W2_cationic_amphipathic` | 阳离子比例与疏水矩百分位秩的均值 |
| `W3_helix_propensity` | 螺旋倾向的库内百分位秩 |
| `W4_hydrophobicity_balance` | `exp(−((GRAVY − 0.2)/0.5)²)` |
| `W5_low_hemolysis` | `1 − hemolysis_prob` |
| `score` | `100 × (0.45·W1 + 0.20·W2 + 0.15·W3 + 0.10·W4 + 0.10·W5)` |

## 5. 解读注意事项

1. **分值是库内相对量**：示例库只有 30 条候选，W1–W3 的百分位秩与完整库不同，因此同一序列在示例与完整库中的得分不可直接比较。
2. **参考集规模小**：示例仅使用 29 条参考肽，其中 15 条进入高活性参考集；这会影响 W1 质心与 W5 模型。完整分析应使用尽可能完备、标注一致的参考集。
3. **排序不是活性预测**：排名靠前只表示在给定参考集与评分体系下更值得优先验证，不代表 MIC、抑菌效果或安全性。
4. **溶血概率为探索性**：示例中溶血模型由 29 条标签训练（5 折交叉验证），不确定性较大；W5 仅作为温和惩罚项。
5. **示例输出不含结构、对接或动力学结果**：这些分析不属于本仓库的四步管线。
