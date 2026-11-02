# DeepSeek-V4.1-Flash 层裁剪工具

把一个完整的 DeepSeek-V4.1-Flash（40 层语言主干 + 3 层 MTP + 32 层视觉塔）裁剪成更少层的模型，
用于低显存环境的推理/框架冒烟测试。裁剪时逐张量复制，**不改动量化格式**（fp8 权重 + fp4 打包专家 +
ue8m0 scale 原样搬运），并同步重写 `config.json` 中所有跟层号相关的字段。

## 文件说明

| 文件 | 作用 |
|---|---|
| `prune_moe_layers_v41.py` | **主脚本**，在原始 `prune_moe_layers.py` 上补齐了 V4.1 的跨层引用重映射、`--dry-run`、`--drop-engram` |
| `prune_moe_layers.py` | 原始脚本（仅支持普通堆叠模型，V4.1 直接跑会产出无法加载的模型，保留作对照） |
| `verify_pruned_v41.py` | 独立复校：逐张量比对 dtype/shape、校验集合与配置，抽样做数据 `torch.equal` |
| `size_report.py` | 只读 safetensors header，统计体积（按命名空间 / 按层 / 最大张量） |

## 环境要求

```bash
export LD_LIBRARY_PATH=/opt/maca-3.10.0/ompi/lib:$LD_LIBRARY_PATH   # 否则 import torch 报 libmpi.so.40
/opt/conda/bin/python -c "import torch, safetensors; print(torch.__version__, safetensors.__version__)"
```

需要 `torch`（2.x，Metax 版即可）、`safetensors`、`tqdm`。源目录必须含 `config.json`
和 `model.safetensors.index.json`；输出目录不能与源目录互相包含。

## 怎么用

### 1. 先 dry-run（几秒钟，强烈建议）

只算层选择、写 `config.json` 和 `layer_mapping.json`，不复制任何权重：

```bash
python prune_moe_layers_v41.py \
  --src /mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V4.1-Flash-1017 \
  --dst /tmp/dryrun --layers 20 --dry-run
```

看输出里的「最终保留主模型层」「重映射跨层引用 …」「输出 NextN/MTP 数量」，确认没报错再正式跑。

### 2. 正式裁剪

```bash
python prune_moe_layers_v41.py \
  --src /mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V4.1-Flash-1017 \
  --dst /path/to/DeepSeek-V4.1-Flash-1017-pruned \
  --layers 20 --shard-size 10
```

跑完日志末尾会出现 `输出模型校验通过` 和 `切分完成： <dst>`，然后给读权限：

```bash
chmod -R a+rX /path/to/DeepSeek-V4.1-Flash-1017-pruned
```

### 3. 两个已验证的配方

| 用途 | 命令关键参数 | 层数 | 体积 | 需要显存 |
|---|---|---|---|---|
| 保留全部能力 | `--layers 20` | 20（40→20） | 337.6 GB | ≥ 8×64GB |
| 低显存冒烟测试 | `--layers 11 --drop-engram` | 11（40→11） | 86.5 GB | 2×72GB（144GB） |

**engram 说明**：40 层里有 2 层（原 1、14 号）各带一张 91.5GB 的哈希表，两张合起来 183GB，
比 2 张 C600-U 的总显存还大。`--drop-engram` 会：① 丢掉只被 engram 引用的整层；
② 把仍然保留的层（如 14 号层，它同时是 kv/index 源层）里的 `engram.*` 张量剔除；
③ 把 `engram_layer_ids` / `engram_num_embeddings` 清空。裁剪后模型不再有 n-gram 记忆能力，
仅适合跑通流程。

### 4. 参数

| 参数 | 说明 |
|---|---|
| `--src` | 源模型目录（必填） |
| `--dst` | 输出目录（必填，不能与源目录互相包含） |
| `--layers` | 目标语言主干层数；不填默认 `ceil(N/2)`。**有下限**：跨层引用锚点层必须保留（V4.1 是 12 层，`--drop-engram` 后是 11 层），填小了会直接报错告诉你合计需要多少层 |
| `--shard-size` | 单个输出 shard 上限（GB），默认 10；内存小或磁盘配额紧时调小（如 4、8） |
| `--overwrite` | 输出目录非空时覆盖 |
| `--dry-run` | 只算配置、不复制权重 |
| `--drop-engram` | 丢弃 engram 层及其 91GB 哈希表（见上） |
| `--skip-validation` | 跳过最终逐 shard 校验（不建议） |

## 输出内容

```
<dst>/
├── config.json                    # 层数、compress_ratios、跨层引用字段全部已重写
├── model.safetensors.index.json   # 新 weight_map
├── model-0000N-of-0000M.safetensors
├── layer_mapping.json             # 源层号 → 新层号，含 cross_layer_config_changes 明细
├── tokenizer.json / tokenizer_config.json / generation_config.json
├── inference/  encoding/          # 从源目录原样复制
└── README.md / LICENSE
```

`layer_mapping.json` 是排障的关键：里面记录了 `kept_source_main_layers`、`main_layer_mapping`、
以及每个跨层字段的旧值→新值。

## 校验

脚本尾部自带校验（`validating shards`）：每个 shard 的实际 key 与 index 逐一对齐、跨 shard 无重复、
`num_hidden_layers` 与逐层数组长度自洽、跨层引用字段全部落在新层号范围内、量化配置未被改动。

需要更强的独立复校（会读源模型 header 逐张量比对）：

```bash
python verify_pruned_v41.py     # 改脚本顶部的 SRC / DST 两个常量
python size_report.py <模型目录>  # 看体积分布，例如确认 engram 表是否已剔除
```

注意：校验只能证明结构和张量搬运正确，**不能证明裁剪后的模型推理质量**；上线前应跑
`inference/convert.py` + `inference/run.sh` 做一次真实推理。