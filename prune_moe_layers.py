#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoE 语言模型及多模态模型的语言层裁剪脚本。

支持范围：
- DeepSeek-V3 / V3.1 / V3.2 / V4 / R1
- GLM MoE/DSA 系列
- Kimi-K2/K2.5/K2.6 等 text_config 嵌套式多模态模型
- MiniMax-M3 / MiniMax-M3-VL（含 FP8、Sparse Attention、独立 MTP）
- BF16、原生 FP8、compressed-tensors W8A8/W4A16

核心规则：
1. 自动从 weight_map 识别层前缀，例如 model.layers.<id>.* 或 layers.<id>.*。
2. num_hidden_layers 只表示主模型层数。
3. 保留前 first_k_dense_replace 个 Dense 层。
4. DeepSeek-V4 额外保留前 num_hash_layers 个 Hash-MoE 层。
5. 根据 num_nextn_predict_layers 保留主模型最后对应数量的层。
6. 从其余主模型层中均匀采样；不指定 --layers 时主模型向上取整切半。
7. 主模型范围之外的额外层单独保留，不计入 --layers。
8. mtp.* / nextn.* 形式的独立权重也会原名保留。
9. 自动同步普通及嵌套逐层配置，包括 MiniMax-M3 的 moe_layer_freq / sparse_attention_config。
10. 自动重映射 quantization_config.modules_to_not_convert 中的语言层编号。
11. MiniMax-M3 的独立 MTP 不按“主干末尾额外层”处理。
12. 不复制 .safetensors_original、index.json_original 等权重备份文件。
"""

import os
import gc
import json
import copy
import shutil
import argparse
import re
from collections import defaultdict

from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file


CONFIG_FILE = "config.json"
INDEX_FILE = "model.safetensors.index.json"
VERSION = "1.0.0"

# 运行时从 weight_map 自动识别。
LAYER_PREFIX = None

KNOWN_LAYER_PREFIXES = (
    # 普通纯文本模型。
    "model.layers.",
    "layers.",
    "transformer.layers.",
    "model.decoder.layers.",
    "decoder.layers.",
    "model.blocks.",
    "blocks.",
    "transformer.h.",
    "model.h.",

    # 多模态模型中的语言模型。
    "language_model.model.layers.",
    "model.language_model.layers.",
    "model.language_model.model.layers.",
    "text_model.model.layers.",
    "model.text_model.layers.",
    "model.text_model.model.layers.",
    "model.llm.layers.",
    "model.llm.model.layers.",
)

# 逐层配置路径。路径相对于真正的语言配置（root / text_config / language_config 等）。
#
# MiniMax-M3 的三个关键逐层字段：
# - moe_layer_freq
# - sparse_attention_config.sparse_disable_index_value
# - sparse_attention_config.sparse_attention_freq
#
# 裁层后必须按照 keep_main 对这些列表做相同采样，否则重编号后的层结构会与
# 实际保留权重不一致。
PER_LAYER_CONFIG_PATHS = (
    ("compress_ratios",),
    ("layer_types",),
    ("mlp_layer_types",),
    ("moe_layer_freq",),
    ("sparse_attention_config", "sparse_disable_index_value"),
    ("sparse_attention_config", "sparse_attention_freq"),
)

# 原脚本已经把这三个字段定义为“必须为列表”；保持原有严格校验。
# MiniMax 新增字段如果在其他模型中恰好是标量，则视为全局配置并原样保留。
STRICT_PER_LAYER_CONFIG_PATHS = {
    ("compress_ratios",),
    ("layer_types",),
    ("mlp_layer_types",),
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


LANGUAGE_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
    "decoder_config",
)


def resolve_language_config(config):
    """
    定位真正描述语言模型结构的配置字典。

    支持：
    1. 纯文本模型：
       config.num_hidden_layers

    2. Kimi 等多模态模型：
       config.text_config.num_hidden_layers

    3. 其他常见嵌套：
       language_config / llm_config / decoder_config

    返回：
        language_config, config_path

    config_path 为 "root" 或 "text_config" 等字段名。
    """
    if not isinstance(config, dict):
        raise TypeError(
            f"config 必须为字典，实际为 {type(config).__name__}"
        )

    if "num_hidden_layers" in config:
        return config, "root"

    for key in LANGUAGE_CONFIG_KEYS:
        nested = config.get(key)

        if (
            isinstance(nested, dict)
            and "num_hidden_layers" in nested
        ):
            return nested, key

    available = [
        key
        for key, value in config.items()
        if isinstance(value, dict)
    ]

    raise KeyError(
        "config.json 中未找到语言模型的 num_hidden_layers。\n"
        "已检查顶层以及："
        + ", ".join(LANGUAGE_CONFIG_KEYS)
        + "\n顶层字典字段："
        + ", ".join(sorted(available))
    )


def remove_name_or_path(config):
    """删除顶层及语言子配置中的旧 _name_or_path。"""
    config.pop("_name_or_path", None)

    language_config, _ = resolve_language_config(config)
    language_config.pop("_name_or_path", None)


def dotted_path(path):
    return ".".join(path)


def get_nested(config, path):
    """读取嵌套配置；路径不存在时返回 None。"""
    current = config

    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


def set_nested(config, path, value):
    """写入已存在的嵌套配置路径。"""
    current = config

    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise TypeError(
                f"配置路径 {dotted_path(path)} 的中间字段 {key} 不是字典。"
            )
        current = child

    current[path[-1]] = value


def is_minimax_m3_model(outer_config, language_config):
    """根据 model_type / architectures 判断是否为 MiniMax-M3 系列。"""
    candidates = [
        outer_config.get("model_type"),
        language_config.get("model_type"),
    ]

    for cfg in (outer_config, language_config):
        architectures = cfg.get("architectures", [])
        if isinstance(architectures, (list, tuple)):
            candidates.extend(architectures)

    normalized = [
        str(item).lower().replace("-", "_")
        for item in candidates
        if item is not None
    ]

    return any(
        "minimax_m3" in item or "minimaxm3" in item
        for item in normalized
    )


def infer_leading_zero_prefix(values, total, field_name):
    """
    从逐层开关列表推断“前置特殊层”数量。

    MiniMax-M3 的 moe_layer_freq / sparse_attention_freq 都是：
        [0, 0, 0, 1, 1, ...]
    因此前 3 层应当作为结构前缀优先保留。

    若列表全为 0，则不认为整个模型都是必须保护的前缀。
    """
    if values is None:
        return 0

    if not isinstance(values, list):
        # 某些模型把同名字段定义为标量（全局频率），这种情况不参与
        # 前置层推断，也不能按层采样。
        return 0

    if len(values) < total:
        raise ValueError(
            f"{field_name} 长度不足：{len(values)} < num_hidden_layers={total}"
        )

    main_values = values[:total]
    count = 0

    for value in main_values:
        if value != 0:
            break
        count += 1

    # 全 0 时不把全部层强制保护。
    if count == total:
        return 0

    return count


def infer_protected_prefix_count(
    language_config,
    total,
    dense_count,
    hash_count,
):
    """综合传统字段与逐层结构字段推断必须保留的前置层。"""
    details = {
        "first_k_dense_replace": dense_count,
        "num_hash_layers": hash_count,
    }

    moe_prefix = infer_leading_zero_prefix(
        language_config.get("moe_layer_freq"),
        total,
        "moe_layer_freq",
    )
    details["moe_layer_freq_leading_zero_prefix"] = moe_prefix

    sparse_config = language_config.get("sparse_attention_config")
    if sparse_config is None:
        sparse_config = {}
    elif not isinstance(sparse_config, dict):
        raise TypeError("sparse_attention_config 应为字典。")

    sparse_freq_prefix = infer_leading_zero_prefix(
        sparse_config.get("sparse_attention_freq"),
        total,
        "sparse_attention_config.sparse_attention_freq",
    )
    details[
        "sparse_attention_freq_leading_zero_prefix"
    ] = sparse_freq_prefix

    sparse_disable_prefix = infer_leading_zero_prefix(
        sparse_config.get("sparse_disable_index_value"),
        total,
        "sparse_attention_config.sparse_disable_index_value",
    )
    details[
        "sparse_disable_index_value_leading_zero_prefix"
    ] = sparse_disable_prefix

    protected = max(details.values())
    return protected, details


def config_has_quant_descriptor(config):
    if not isinstance(config, dict):
        return False
    return bool(
        config.get("quantization_config")
        or config.get("compression_config")
    )


def detect_quant_kind(outer_config, language_config):
    """
    同时支持量化配置位于顶层或语言子配置。

    MiniMax-M3-FP8 的 quantization_config 位于顶层；
    一些 Kimi/DeepSeek checkpoint 则可能位于语言配置。
    """
    candidates = [("root", outer_config)]
    if language_config is not outer_config:
        candidates.append(("language_config", language_config))

    for path, cfg in candidates:
        if config_has_quant_descriptor(cfg):
            print(f"量化配置位置：{path}")
            return check_quant_config(cfg), path

    print(
        "未检测到 quantization_config 或 "
        "compression_config，按普通 checkpoint 处理。"
    )
    return "none", "none"


def iter_config_scopes(outer_config, language_config):
    """遍历顶层和语言配置，避免 root 模型重复处理同一个 dict。"""
    yield "root", outer_config
    if language_config is not outer_config:
        yield "language_config", language_config


def remap_modules_to_not_convert(
    outer_config,
    language_config,
    layer_mapping,
    total_source_layers,
):
    """
    重映射 quantization_config.modules_to_not_convert 中的语言层编号。

    - 保留的层：old_id -> new_id；
    - 被裁掉且确实属于主层/额外层编号空间的规则：删除；
    - vision / projector / lm_head 等非语言层规则：原样保留。
    """
    changes = []
    pattern = re.compile(
        r"^" + re.escape(LAYER_PREFIX) + r"(\d+)(\..+)$"
    )

    for scope_name, cfg in iter_config_scopes(
        outer_config, language_config
    ):
        quant = cfg.get("quantization_config")
        if not isinstance(quant, dict):
            continue

        modules = quant.get("modules_to_not_convert")
        if modules is None:
            continue
        if not isinstance(modules, list):
            raise TypeError(
                f"{scope_name}.quantization_config.modules_to_not_convert "
                "应为列表。"
            )

        new_modules = []
        remapped = 0
        removed = 0

        for name in modules:
            if not isinstance(name, str):
                new_modules.append(name)
                continue

            match = pattern.match(name)
            if match is None:
                new_modules.append(name)
                continue

            old_id = int(match.group(1))
            suffix = match.group(2)

            if old_id in layer_mapping:
                new_id = layer_mapping[old_id]
                new_name = f"{LAYER_PREFIX}{new_id}{suffix}"
                new_modules.append(new_name)
                if new_name != name:
                    remapped += 1
                continue

            # 对属于语言层编号空间但已被裁掉的规则直接删除。
            # total_source_layers 之外的编号如果未被识别为 keep_extra，也同样不能
            # 留在新 checkpoint 里，否则会指向不存在的语言层。
            removed += 1

        new_modules = list(dict.fromkeys(new_modules))
        quant["modules_to_not_convert"] = new_modules

        change = {
            "scope": scope_name,
            "source_count": len(modules),
            "target_count": len(new_modules),
            "remapped_count": remapped,
            "removed_count": removed,
            "source_num_hidden_layers": total_source_layers,
        }
        changes.append(change)

        print(
            "更新 modules_to_not_convert："
            f"{scope_name}，{len(modules)} -> {len(new_modules)}，"
            f"重映射 {remapped}，删除 {removed}"
        )

    return changes


def quant_core_snapshot(outer_config, language_config):
    """
    保存量化核心配置快照。modules_to_not_convert 允许因裁层而改变，
    其他量化参数必须保持一致。
    """
    snapshot = {}

    for scope_name, cfg in iter_config_scopes(
        outer_config, language_config
    ):
        for field in ("compression_config", "quantization_config"):
            value = cfg.get(field)
            if value is None:
                continue

            value = copy.deepcopy(value)
            if field == "quantization_config" and isinstance(value, dict):
                value.pop("modules_to_not_convert", None)

            snapshot[f"{scope_name}.{field}"] = value

    return snapshot


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def ids_for_prefix(weight_map, prefix):
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.")
    ids = set()
    matched_names = 0

    for name in weight_map:
        match = pattern.match(name)
        if match:
            ids.add(int(match.group(1)))
            matched_names += 1

    return ids, matched_names


def detect_layer_prefix(weight_map, total_layers):
    """
    自动识别主模型层权重前缀。

    支持常见形式：
        model.layers.0.*
        layers.0.*
        transformer.layers.0.*
        model.blocks.0.*
    """
    expected = set(range(total_layers))
    candidates = []

    for priority, prefix in enumerate(KNOWN_LAYER_PREFIXES):
        ids, matched_names = ids_for_prefix(weight_map, prefix)
        if expected.issubset(ids):
            candidates.append(
                (0, priority, -matched_names, prefix, ids)
            )

    # 通用回退：只接受 layers / blocks / h 这种结构段，
    # 避免把 experts.<id> 误识别成模型层。
    generic_prefixes = set()
    generic_pattern = re.compile(
        r"^(.+?(?:layers|blocks|h)\.)(\d+)\."
    )

    for name in weight_map:
        match = generic_pattern.match(name)
        if match:
            generic_prefixes.add(match.group(1))

    for prefix in sorted(generic_prefixes):
        if prefix in KNOWN_LAYER_PREFIXES:
            continue

        ids, matched_names = ids_for_prefix(weight_map, prefix)
        if expected.issubset(ids):
            candidates.append(
                (1, len(prefix), -matched_names, prefix, ids)
            )

    if not candidates:
        examples = sorted(weight_map)[:30]
        raise ValueError(
            "无法从 weight_map 识别模型层前缀。\n"
            f"config.num_hidden_layers={total_layers}\n"
            "前 30 个 tensor 名：\n  "
            + "\n  ".join(examples)
        )

    candidates.sort(key=lambda item: item[:3])
    _, _, _, prefix, found_ids = candidates[0]

    print(f"识别到层权重前缀：{prefix}<layer_id>.*")
    print(
        f"识别到层编号范围：{min(found_ids)}~{max(found_ids)}，"
        f"共 {len(found_ids)} 个编号"
    )

    return prefix


def layer_id(name):
    if LAYER_PREFIX is None:
        raise RuntimeError("尚未初始化 LAYER_PREFIX。")

    if not name.startswith(LAYER_PREFIX):
        return None

    remainder = name[len(LAYER_PREFIX):]
    number, separator, _ = remainder.partition(".")

    if not separator:
        raise ValueError(f"无法解析层权重名称：{name}")

    try:
        return int(number)
    except ValueError as exc:
        raise ValueError(f"层编号不是整数：{name}") from exc


def rename_layer(name, new_id):
    old_id = layer_id(name)

    if old_id is None:
        raise ValueError(f"不是当前层前缀下的权重：{name}")

    remainder = name[len(LAYER_PREFIX):]
    _, separator, suffix = remainder.partition(".")

    if not separator:
        raise ValueError(f"无法重命名层权重：{name}")

    return f"{LAYER_PREFIX}{new_id}.{suffix}"

def evenly_sample(candidates, count):
    candidates = list(candidates)

    if count < 0 or count > len(candidates):
        raise ValueError(
            f"采样数量非法：需要 {count}，候选只有 {len(candidates)}"
        )

    if count == 0:
        return []

    if count == len(candidates):
        return candidates

    if count == 1:
        return [candidates[len(candidates) // 2]]

    last = len(candidates) - 1
    positions = [i * last // (count - 1) for i in range(count)]
    sampled = [candidates[pos] for pos in positions]

    if len(set(sampled)) != count:
        raise RuntimeError(f"均匀采样出现重复层：{sampled}")

    return sampled


def choose_layers(total, target, protected_prefix_count, tail_count):
    """
    选择主模型层：
    - 强制保留前 protected_prefix_count 层；
    - 强制保留主模型最后 tail_count 层；
    - 从中间均匀采样。
    """
    if total <= 0:
        raise ValueError(f"num_hidden_layers 非法：{total}")

    if target <= 0 or target > total:
        raise ValueError(
            f"目标层数必须在 1~{total} 之间，当前为 {target}"
        )

    if (
        protected_prefix_count < 0
        or protected_prefix_count > total
    ):
        raise ValueError(
            "前置结构层数量非法："
            f"{protected_prefix_count}"
        )

    if tail_count < 0 or tail_count > total:
        raise ValueError(
            f"num_nextn_predict_layers 非法：{tail_count}"
        )

    prefix_ids = list(range(protected_prefix_count))
    tail_ids = (
        list(range(total - tail_count, total))
        if tail_count
        else []
    )

    mandatory = sorted(set(prefix_ids) | set(tail_ids))

    if len(mandatory) > target:
        raise ValueError(
            "必须保留的前置结构层和末尾层超过目标层数：\n"
            f"前置层={prefix_ids}\n"
            f"末尾层={tail_ids}\n"
            f"目标={target}"
        )

    mandatory_set = set(mandatory)
    middle_candidates = [
        old_id
        for old_id in range(total)
        if old_id not in mandatory_set
    ]

    sampled_middle = evenly_sample(
        middle_candidates,
        target - len(mandatory),
    )

    keep = sorted(mandatory + sampled_middle)

    if len(keep) != target or len(set(keep)) != target:
        raise RuntimeError(f"最终层选择结果异常：{keep}")

    return keep, prefix_ids, sampled_middle, tail_ids

def group_by_shard(weight_map):
    grouped = defaultdict(list)

    for name, filename in weight_map.items():
        grouped[filename].append(name)

    for names in grouped.values():
        names.sort()

    return grouped


def repair_weight_map_from_shards(src, weight_map):
    """
    校验 model.safetensors.index.json 中的 weight_map，并在 shard 落盘位置
    与 index 不一致时自动修正。

    只扫描 index 实际引用的 safetensors 文件，不读取 tensor 数据，
    safe_open(...).keys() 只读取 safetensors header，因此开销远小于加载权重。

    处理规则：
    1. index 中 tensor 在声明 shard 中存在：保持不变；
    2. index 中 tensor 不在声明 shard、但在其他已引用 shard 中唯一存在：
       自动把 weight_map 修正到真实 shard；
    3. 同名 tensor 出现在多个 shard：报错，避免静默选择错误权重；
    4. index 引用的 tensor 在所有 shard 中都不存在：报错，说明源 checkpoint
       本身不完整或 index 与实际文件集合严重不一致；绝不伪造量化 scale。
    5. shard 中存在但 index 未登记的 tensor：补入修正后的 weight_map，避免
       stale index 漏掉实际权重。
    """
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("weight_map 为空或格式非法。")

    indexed_files = sorted(set(weight_map.values()))

    # 正常 HuggingFace shard 文件。即使 stale index 没有引用到某个 shard，
    # 也扫描它，以便找回被错误指到其他 shard 的 tensor。
    shard_pattern = re.compile(
        r"^model-\d+-of-\d+\.safetensors$"
    )
    discovered_model_shards = sorted(
        entry.name
        for entry in os.scandir(src)
        if entry.is_file() and shard_pattern.match(entry.name)
    )

    candidate_files = sorted(
        set(indexed_files) | set(discovered_model_shards)
    )
    actual_locations = defaultdict(list)

    print("\n校验源 checkpoint index 与 safetensors shard...")
    print(f"  index tensor 数：{len(weight_map)}")
    print(f"  index shard 数：{len(indexed_files)}")
    print(f"  实际扫描 shard 数：{len(candidate_files)}")

    for filename in tqdm(candidate_files, desc="scanning source shard headers"):
        path = os.path.join(src, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"index 引用了不存在的源 shard：{path}"
            )

        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                actual_locations[name].append(filename)

    duplicate_locations = {
        name: files
        for name, files in actual_locations.items()
        if len(files) > 1
    }

    if duplicate_locations:
        examples = list(sorted(duplicate_locations.items()))[:20]
        raise ValueError(
            "源 checkpoint 中同名 tensor 出现在多个 shard，无法安全修复 index：\n  "
            + "\n  ".join(
                f"{name}: {files}" for name, files in examples
            )
        )

    actual_map = {
        name: files[0]
        for name, files in actual_locations.items()
    }

    indexed_names = set(weight_map)
    actual_names = set(actual_map)

    missing_everywhere = sorted(indexed_names - actual_names)

    # DeepSeek-V4 的 wo_a 有一类常见 stale-index 情况：
    # index 中残留 `<prefix>.attn.wo_a.scale`，但真实 shard 中 wo_a
    # 可能是 BF16 权重，因此根本不需要 weight scale；也可能 FP8 scale
    # 使用 `.scale_inv` / `.weight_scale` / `.weight_scale_inv` 等实际名称。
    #
    # 这里不能无脑忽略缺失 scale：
    # - BF16/FP16/FP32 wo_a.weight：允许删除 stale `.scale` index 项；
    # - FP8 wo_a.weight：必须能在真实 shard 中找到另一种 scale tensor，
    #   否则仍视为 checkpoint 不完整并报错。
    tolerated_stale = []
    fatal_missing = []

    def _actual_tensor_dtype(name):
        filename = actual_map.get(name)
        if filename is None:
            return None
        path = os.path.join(src, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            return str(f.get_slice(name).get_dtype())

    for missing_name in missing_everywhere:
        match = re.match(r"^(.+\.attn\.wo_a)\.scale$", missing_name)

        if not match:
            fatal_missing.append(missing_name)
            continue

        base = match.group(1)
        weight_name = base + ".weight"

        if weight_name not in actual_map:
            fatal_missing.append(missing_name)
            continue

        alt_scale_names = [
            base + ".scale_inv",
            base + ".weight_scale",
            base + ".weight_scale_inv",
        ]
        actual_alt_scales = [
            name for name in alt_scale_names if name in actual_map
        ]

        weight_dtype = _actual_tensor_dtype(weight_name)
        dtype_upper = (weight_dtype or "").upper()
        is_fp8 = ("F8" in dtype_upper or "FLOAT8" in dtype_upper)

        if actual_alt_scales:
            tolerated_stale.append(
                (missing_name, weight_name, weight_dtype, actual_alt_scales)
            )
            continue

        if not is_fp8:
            tolerated_stale.append(
                (missing_name, weight_name, weight_dtype, [])
            )
            continue

        fatal_missing.append(missing_name)

    if tolerated_stale:
        print(
            "警告：发现 DeepSeek-V4 wo_a 的 stale `.scale` index 项："
            f"{len(tolerated_stale)} 个；已根据真实权重 dtype/scale 命名安全忽略。"
        )
        for missing_name, weight_name, weight_dtype, alt_scales in tolerated_stale[:50]:
            print(f"  stale index: {missing_name}")
            print(f"    weight    : {weight_name}")
            print(f"    dtype     : {weight_dtype}")
            if alt_scales:
                print(f"    real scale: {alt_scales}")
            else:
                print("    real scale: none (non-FP8 wo_a weight)")
        if len(tolerated_stale) > 50:
            print("  ...")

    if fatal_missing:
        raise ValueError(
            "源 index 引用了实际所有 shard 中都不存在、且无法安全判定为 "
            "DeepSeek-V4 wo_a stale scale 的 tensor。\n"
            f"缺失数量：{len(fatal_missing)}\n"
            "前 50 个缺失 tensor：\n  "
            + "\n  ".join(fatal_missing[:50])
            + "\n提示：如果缺失项是 FP8 wo_a.scale，而真实 shard 中也没有 "
              "scale_inv/weight_scale/weight_scale_inv，则不能安全跳过。"
        )

    # 注意：最终返回 actual_map，而不是原 weight_map。
    # 因此 tolerated stale index 项自然被删除，真实 shard 中未登记的
    # alternate scale（如 weight_scale_inv）则会被保留下来。

    tolerated_stale_names = {
        item[0] for item in tolerated_stale
    }

    relocated = []
    for name, indexed_filename in weight_map.items():
        # 已确认可安全忽略的 stale index 项并不存在于真实 shard。
        # 它们应从修正后的 weight_map 中删除，不能再访问 actual_map[name]。
        if name in tolerated_stale_names:
            continue

        actual_filename = actual_map.get(name)
        if actual_filename is None:
            # 理论上前面的 fatal_missing 已覆盖所有非容忍缺失项。
            # 这里保留防御性检查，避免再次出现裸 KeyError。
            raise ValueError(
                f"内部错误：tensor 在真实 shard 中不存在但未被缺失检查捕获：{name}"
            )

        if actual_filename != indexed_filename:
            relocated.append(
                (name, indexed_filename, actual_filename)
            )

    extra_actual = sorted(actual_names - indexed_names)

    if relocated:
        print(
            f"警告：发现 {len(relocated)} 个 tensor 的 index shard 位置错误，"
            "将按 safetensors 实际位置自动修正。"
        )
        for name, old_file, new_file in relocated[:20]:
            print(f"  {name}")
            print(f"    index : {old_file}")
            print(f"    actual: {new_file}")
        if len(relocated) > 20:
            print("  ...")

    if extra_actual:
        print(
            f"警告：发现 {len(extra_actual)} 个 tensor 存在于 shard，"
            "但未登记在 index；将补入修正后的 weight_map。"
        )
        for name in extra_actual[:20]:
            print(f"  {name} -> {actual_map[name]}")
        if len(extra_actual) > 20:
            print("  ...")

    if not relocated and not extra_actual and not tolerated_stale:
        print("源 checkpoint index 与 shard 内容一致。")
    else:
        print(
            f"修正后 tensor 数：{len(actual_map)}；"
            f"删除 stale index：{len(tolerated_stale)}；"
            f"重定位：{len(relocated)}；补录：{len(extra_actual)}"
        )

    return actual_map


def check_quant_config(config):
    """
    识别语言模型量化格式。

    支持：
    - DeepSeek 原生 FP8 quantization_config；
    - compressed-tensors quantization_config，例如 Kimi W4A16；
    - compressed-tensors compression_config，例如 W8A8。

    返回：
        "fp8"、"w8a8"、"compressed_tensors"、"other" 或 "none"
    """
    quantization = config.get(
        "quantization_config"
    )

    if quantization:
        quant_method = str(
            quantization.get(
                "quant_method", ""
            )
        ).lower().replace("_", "-")

        print("检测到 quantization_config：")
        print("  quant_method:", quant_method)
        print(
            "  format:",
            quantization.get("format"),
        )
        print(
            "  fmt:",
            quantization.get("fmt"),
        )
        print(
            "  scale_fmt:",
            quantization.get("scale_fmt"),
        )
        print(
            "  weight_block_size:",
            quantization.get(
                "weight_block_size"
            ),
        )

        if quant_method == "fp8":
            return "fp8"

        if quant_method == "compressed-tensors":
            for group_name, group in (
                quantization.get(
                    "config_groups", {}
                ).items()
            ):
                weights = (
                    group.get("weights") or {}
                )
                activations = (
                    group.get(
                        "input_activations"
                    )
                    or {}
                )

                print(
                    f"  {group_name}: "
                    f"W{weights.get('num_bits')}"
                    f"A{activations.get('num_bits')}"
                )

            return "compressed_tensors"

        print(
            "警告：检测到未知 quantization_config；"
            "仍将原样保留配置和 tensor。"
        )
        return "other"

    compression = config.get(
        "compression_config"
    )

    if compression:
        method = str(
            compression.get(
                "quant_method", ""
            )
        ).lower().replace("_", "-")

        print("检测到 compression_config：")
        print("  quant_method:", method)
        print(
            "  format:",
            compression.get("format"),
        )

        is_w8a8 = False

        for group_name, group in (
            compression.get(
                "config_groups", {}
            ).items()
        ):
            weights = group.get(
                "weights", {}
            ) or {}

            activations = group.get(
                "input_activations", {}
            ) or {}

            weight_bits = weights.get(
                "num_bits"
            )
            activation_bits = (
                activations.get("num_bits")
            )

            print(
                f"  {group_name}: "
                f"W{weight_bits}A{activation_bits}"
            )

            if (
                weight_bits == 8
                and activation_bits == 8
            ):
                is_w8a8 = True

        if is_w8a8:
            return "w8a8"

        if method == "compressed-tensors":
            return "compressed_tensors"

        return "other"

    print(
        "未检测到 quantization_config 或 "
        "compression_config，按普通 checkpoint 处理。"
    )
    return "none"

def check_auto_map(src, *configs):
    """
    检查顶层配置和语言子配置中的 auto_map。
    文件不存在只警告，不中止切分。
    """
    missing = []

    for config in configs:
        if not isinstance(config, dict):
            continue

        for target in config.get(
            "auto_map", {}
        ).values():
            if not isinstance(target, str):
                continue

            module_file = (
                target.split(".")[0] + ".py"
            )

            path = os.path.join(
                src,
                module_file,
            )

            if not os.path.isfile(path):
                missing.append(path)

    if missing:
        print("\n警告：auto_map 引用的文件不存在：")

        for path in sorted(set(missing)):
            print(" ", path)

        print(
            "该警告不阻止权重切分；"
            "但使用 trust_remote_code 加载时应确认模型代码齐全。"
        )

def find_standalone_nextn_tensors(weight_map):
    """
    查找不属于主层前缀、但名称中含 mtp / nextn 的独立权重。
    这些 tensor 在 build_name_map 中会按原名保留。
    """
    pattern = re.compile(
        r"(^|\.)(mtp|nextn|next_n|nextn_predict)(\.|_|$)",
        re.IGNORECASE,
    )

    return sorted(
        name
        for name in weight_map
        if layer_id(name) is None and pattern.search(name)
    )


def update_per_layer_config(
    config,
    keep_main,
    total_main,
    source_nextn_count,
    output_nextn_count,
):
    """
    同步普通及嵌套逐层配置。

    支持两种源格式：
    - 列表长度 == num_hidden_layers：只描述主模型层；
    - 列表长度 == num_hidden_layers + num_nextn_predict_layers：
      同时描述主模型层和额外 NextN/MTP 层。
    """
    changes = {}

    for path in PER_LAYER_CONFIG_PATHS:
        values = get_nested(config, path)

        if values is None:
            continue

        field_name = dotted_path(path)

        if not isinstance(values, list):
            if path in STRICT_PER_LAYER_CONFIG_PATHS:
                raise TypeError(
                    f"config.{field_name} 应为列表，"
                    f"实际为 {type(values).__name__}"
                )

            print(
                f"逐层候选字段 {field_name} 为标量/非列表，"
                "按全局配置原样保留。"
            )
            continue

        old_length = len(values)

        if old_length == total_main:
            new_values = [values[old_id] for old_id in keep_main]

        elif old_length == total_main + source_nextn_count:
            main_values = [values[old_id] for old_id in keep_main]
            nextn_values = values[
                total_main:
                total_main + output_nextn_count
            ]
            new_values = main_values + nextn_values

        else:
            raise ValueError(
                f"config.{field_name} 长度无法解释：{old_length}\n"
                f"期望 {total_main}（只含主层）或 "
                f"{total_main + source_nextn_count}（含 NextN/MTP）。"
            )

        set_nested(config, path, new_values)
        changes[field_name] = {
            "source_length": old_length,
            "target_length": len(new_values),
        }

        print(
            f"更新逐层配置 {field_name}："
            f"{old_length} -> {len(new_values)}"
        )

    return changes


def choose_extra_layers(weight_map, total, configured_count):
    """选择主模型范围之外的额外 NextN/MTP 层。"""
    if configured_count <= 0:
        return [], []

    found_ids = {
        layer_id(name)
        for name in weight_map
        if layer_id(name) is not None
    }

    extra_layers = sorted(
        old_id for old_id in found_ids
        if old_id >= total
    )

    standard_layers = list(
        range(total, total + configured_count)
    )

    if all(old_id in found_ids for old_id in standard_layers):
        keep_extra = standard_layers
    elif len(extra_layers) >= configured_count:
        keep_extra = extra_layers[:configured_count]
        print("\n警告：额外层没有完全采用标准编号。")
        print(f"标准编号应为：{standard_layers}")
        print(f"实际额外层：{extra_layers}")
        print(f"将保留额外层：{keep_extra}")
    else:
        keep_extra = extra_layers
        print(
            "\n警告：config 声明 "
            f"{configured_count} 个 NextN/MTP 层，"
            f"但 checkpoint 只找到 {len(keep_extra)} 个额外层。"
        )
        print(f"实际额外层：{extra_layers}")

    ignored_extra = [
        old_id for old_id in extra_layers
        if old_id not in set(keep_extra)
    ]

    return keep_extra, ignored_extra


def build_name_map(weight_map, total, keep_main, keep_extra):
    """分别重编号主模型层和额外 NextN/MTP 层。"""
    main_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(keep_main)
    }

    extra_mapping = {
        old_id: len(keep_main) + offset
        for offset, old_id in enumerate(keep_extra)
    }

    found_main = set()

    for name in weight_map:
        old_id = layer_id(name)
        if old_id is not None and 0 <= old_id < total:
            found_main.add(old_id)

    missing = set(range(total)) - found_main

    if missing:
        raise ValueError(
            f"源 checkpoint 缺少 config 声明的主模型层：{sorted(missing)}"
        )

    name_map = {}

    for old_name in weight_map:
        old_id = layer_id(old_name)

        if old_id is None:
            name_map[old_name] = old_name
        elif old_id in main_mapping:
            name_map[old_name] = rename_layer(
                old_name, main_mapping[old_id]
            )
        elif old_id in extra_mapping:
            name_map[old_name] = rename_layer(
                old_name, extra_mapping[old_id]
            )

    if len(name_map) != len(set(name_map.values())):
        raise ValueError("重命名后出现重复 tensor 名。")

    return name_map, main_mapping, extra_mapping

def check_scale_pairs(source_weight_map, name_map):
    """
    检查量化权重及其 scale 是否同时被保留。

    支持：
        module.weight + module.scale              # DeepSeek-V4 原生 FP8
        module.weight + module.scale_inv
        module.weight + module.weight_scale
        module.weight + module.weight_scale_inv
        module.weight_packed + module.weight_scale

    对 `.scale` / `.scale_inv` 仅在同 base 下确实存在权重 tensor 时才视作
    weight-scale 配对，避免把其他用途的 scale 误判为量化 scale。
    """
    source_names = set(source_weight_map)
    kept_names = set(name_map)
    output_names = set(name_map.values())

    weight_suffixes = (
        ".weight",
        ".weight_packed",
        ".packed_weight",
        ".weight_compressed",
    )

    scale_suffixes = (
        ".scale",
        ".scale_inv",
        ".weight_scale",
        ".weight_scale_inv",
    )

    pair_counts = {
        suffix: 0
        for suffix in scale_suffixes
    }

    unrecognized_scales = []

    for old_name in kept_names:
        matched_scale_suffix = next(
            (
                suffix
                for suffix in scale_suffixes
                if old_name.endswith(suffix)
            ),
            None,
        )

        if matched_scale_suffix is None:
            continue

        base = old_name[:-len(matched_scale_suffix)]

        source_weights = [
            base + suffix
            for suffix in weight_suffixes
            if base + suffix in source_names
        ]

        if not source_weights:
            # 这类 scale 可能不是 weight scale，例如单独的激活/归一化 scale。
            unrecognized_scales.append(old_name)
            continue

        if not any(
            weight_name in kept_names
            for weight_name in source_weights
        ):
            raise ValueError(
                "保留了 scale，但丢失了对应量化权重："
                f"{old_name}，候选权重={source_weights}"
            )

        new_scale = name_map[old_name]

        if new_scale not in output_names:
            raise ValueError(
                f"重命名后缺少 scale：{new_scale}"
            )

        pair_counts[matched_scale_suffix] += 1

    # 反向检查：如果源 checkpoint 里同 base 的 scale 存在，选中 weight 后
    # 对应 scale 不能被裁层逻辑单独丢失。
    for old_name in kept_names:
        matched_weight_suffix = next(
            (
                suffix
                for suffix in weight_suffixes
                if old_name.endswith(suffix)
            ),
            None,
        )

        if matched_weight_suffix is None:
            continue

        base = old_name[:-len(matched_weight_suffix)]

        for scale_suffix in scale_suffixes:
            old_scale = base + scale_suffix

            if (
                old_scale in source_names
                and old_scale not in kept_names
            ):
                raise ValueError(
                    "保留了量化权重，但丢失了 scale："
                    f"{old_name} -> {old_scale}"
                )

    print("量化 weight/scale 配对检查通过：")
    for suffix in scale_suffixes:
        print(f"  {suffix}: {pair_counts[suffix]} 组")

    if unrecognized_scales:
        print(
            "提示：发现无法与同 base 权重匹配的 scale tensor："
            f"{len(unrecognized_scales)} 个；"
            "它们仍按原裁层规则保留，不作为 weight-scale 配对统计。"
        )

def is_weight_artifact(name):
    """
    判断文件是否属于模型权重、权重索引或其备份文件。

    会排除例如：
        model-00001-of-00135.safetensors
        model-00135-of-00135.safetensors_original
        model.safetensors.index.json
        model.safetensors.index.json_original
        pytorch_model.bin
        pytorch_model.bin.bak
    """
    lower_name = name.lower()

    if ".safetensors" in lower_name:
        return True

    if lower_name.startswith(
        "model.safetensors.index.json"
    ):
        return True

    if lower_name.startswith("pytorch_model"):
        return True

    if lower_name.endswith(
        (
            ".bin",
            ".bin_original",
            ".bin.bak",
            ".pt",
            ".pt_original",
            ".pt.bak",
            ".pth",
            ".pth_original",
            ".pth.bak",
            ".ckpt",
            ".ckpt_original",
            ".ckpt.bak",
        )
    ):
        return True

    return False


def copy_aux_files(src, dst):
    """
    复制模型辅助文件和辅助目录。

    会复制：
        tokenizer、generation_config、configuration*.json、
        README、assets、encoding、inference 等。

    不复制：
        原始权重、权重索引、对应备份文件、缓存目录。
    """
    skip_names = {
        CONFIG_FILE,
        INDEX_FILE,
        ".git",
        ".cache",
        "__pycache__",
    }

    for entry in os.scandir(src):
        name = entry.name

        if name in skip_names:
            continue

        if entry.is_file():
            if is_weight_artifact(name):
                continue

            shutil.copy2(
                entry.path,
                os.path.join(dst, name),
            )

        elif entry.is_dir():
            shutil.copytree(
                entry.path,
                os.path.join(dst, name),
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".cache",
                    "__pycache__",
                    "*.safetensors",
                    "*.safetensors_*",
                    "*.safetensors.*",
                    "model.safetensors.index.json*",
                    "pytorch_model*",
                    "*.bin",
                    "*.bin_*",
                    "*.bin.*",
                    "*.pt",
                    "*.pt_*",
                    "*.pt.*",
                    "*.pth",
                    "*.pth_*",
                    "*.pth.*",
                    "*.ckpt",
                    "*.ckpt_*",
                    "*.ckpt.*",
                ),
            )



class ShardWriter:
    def __init__(self, dst, max_size):
        if max_size <= 0:
            raise ValueError("shard_size 必须大于 0。")

        self.dst = dst
        self.max_size = max_size
        self.current = {}
        self.current_size = 0
        self.total_size = 0
        self.temp_files = []
        self.temp_map = {}
        self.output_names = set()

    def add(self, name, tensor):
        if name in self.output_names:
            raise ValueError(f"输出 tensor 名重复：{name}")

        size = tensor_bytes(tensor)

        if self.current and self.current_size + size > self.max_size:
            self.flush()

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self.current[name] = tensor
        self.current_size += size
        self.total_size += size
        self.output_names.add(name)

    def flush(self):
        if not self.current:
            return

        shard_id = len(self.temp_files) + 1
        temp_name = f".tmp-shard-{shard_id:05d}.safetensors"

        save_file(
            self.current,
            os.path.join(self.dst, temp_name),
            metadata={"format": "pt"},
        )

        for name in self.current:
            self.temp_map[name] = temp_name

        print(
            f"写入临时 shard {shard_id}："
            f"{len(self.current)} tensors，"
            f"{self.current_size / 1024**3:.2f} GB"
        )

        self.temp_files.append(temp_name)
        self.current = {}
        self.current_size = 0
        gc.collect()

    def finish(self):
        self.flush()
        shard_count = len(self.temp_files)

        if shard_count == 0:
            raise RuntimeError("没有写入任何 tensor。")

        temp_to_final = {}

        for index, temp_name in enumerate(self.temp_files, start=1):
            final_name = (
                f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            )

            os.replace(
                os.path.join(self.dst, temp_name),
                os.path.join(self.dst, final_name),
            )

            temp_to_final[temp_name] = final_name

        weight_map = {
            name: temp_to_final[temp_name]
            for name, temp_name in self.temp_map.items()
        }

        return weight_map, self.total_size


def validate_output(
    dst,
    target_main,
    output_extra_layer_count,
    output_nextn_count,
):
    outer_config = read_json(
        os.path.join(dst, CONFIG_FILE)
    )

    language_config, config_path = (
        resolve_language_config(
            outer_config
        )
    )

    index = read_json(
        os.path.join(dst, INDEX_FILE)
    )
    weight_map = index["weight_map"]

    if (
        int(
            language_config[
                "num_hidden_layers"
            ]
        )
        != target_main
    ):
        raise ValueError(
            "输出语言配置中的 "
            "num_hidden_layers 不正确。"
        )

    actual_nextn = int(
        language_config.get(
            "num_nextn_predict_layers",
            0,
        )
        or 0
    )

    if actual_nextn != output_nextn_count:
        raise ValueError(
            "输出语言配置中的 "
            "num_nextn_predict_layers 不正确："
            f"期望 {output_nextn_count}，"
            f"实际 {actual_nextn}"
        )

    actual_layers = {
        layer_id(name)
        for name in weight_map
        if layer_id(name) is not None
    }

    expected_layers = set(
        range(
            target_main
            + output_extra_layer_count
        )
    )

    if actual_layers != expected_layers:
        raise ValueError(
            f"输出语言层编号异常："
            f"期望 {sorted(expected_layers)}，"
            f"实际 {sorted(actual_layers)}"
        )

    for path in PER_LAYER_CONFIG_PATHS:
        values = get_nested(language_config, path)

        if values is None:
            continue

        field_name = dotted_path(path)
        allowed_lengths = {
            target_main,
            target_main + output_nextn_count,
        }

        if not isinstance(values, list):
            if path in STRICT_PER_LAYER_CONFIG_PATHS:
                raise TypeError(
                    f"输出 {config_path}.{field_name} 应为列表。"
                )
            continue

        if len(values) not in allowed_lengths:
            raise ValueError(
                f"输出 {config_path}.{field_name} "
                f"长度异常：{len(values)}，"
                f"允许 {sorted(allowed_lengths)}"
            )

    # quantization_config.modules_to_not_convert 不能再引用不存在的语言层。
    valid_layer_ids = set(
        range(target_main + output_extra_layer_count)
    )
    module_pattern = re.compile(
        r"^" + re.escape(LAYER_PREFIX) + r"(\d+)(?:\.|$)"
    )

    for scope_name, cfg in iter_config_scopes(
        outer_config, language_config
    ):
        quant = cfg.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        modules = quant.get("modules_to_not_convert")
        if not isinstance(modules, list):
            continue

        invalid = []
        for name in modules:
            if not isinstance(name, str):
                continue
            match = module_pattern.match(name)
            if match and int(match.group(1)) not in valid_layer_ids:
                invalid.append(name)

        if invalid:
            raise ValueError(
                f"输出 {scope_name}.quantization_config.modules_to_not_convert "
                "仍引用不存在的语言层：\n  "
                + "\n  ".join(invalid[:30])
            )

    grouped = group_by_shard(weight_map)
    all_names = set()

    for filename, expected_names in tqdm(
        sorted(grouped.items()),
        desc="validating shards",
    ):
        path = os.path.join(dst, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"输出 shard 不存在：{path}"
            )

        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as f:
            actual_names = set(f.keys())

        expected_names = set(expected_names)

        if actual_names != expected_names:
            raise ValueError(
                f"{filename} 与 index 内容不一致。\n"
                f"missing="
                f"{sorted(expected_names - actual_names)[:20]}\n"
                f"extra="
                f"{sorted(actual_names - expected_names)[:20]}"
            )

        duplicate = all_names & actual_names

        if duplicate:
            raise ValueError(
                "不同 shard 中存在重复 tensor："
                f"{sorted(duplicate)[:20]}"
            )

        all_names.update(actual_names)

    print(
        f"输出模型校验通过；"
        f"语言配置位置：{config_path}"
    )

def prepare_dirs(src, dst, overwrite):
    src = os.path.realpath(os.path.abspath(src))
    dst = os.path.realpath(os.path.abspath(dst))

    if not os.path.isdir(src):
        raise FileNotFoundError(f"源模型目录不存在：{src}")

    if src == dst:
        raise ValueError("src 和 dst 不能相同。")

    common = os.path.commonpath([src, dst])

    if common == src or common == dst:
        raise ValueError("src 和 dst 不能互相包含。")

    if os.path.exists(dst):
        if overwrite:
            shutil.rmtree(dst)
        elif os.listdir(dst):
            raise FileExistsError(
                f"输出目录非空：{dst}，请添加 --overwrite"
            )

    os.makedirs(dst, exist_ok=True)
    return src, dst


def prune(src, dst, requested_layers, shard_size, overwrite, skip_validation):
    global LAYER_PREFIX

    src, dst = prepare_dirs(src, dst, overwrite)

    config_path = os.path.join(src, CONFIG_FILE)
    index_path = os.path.join(src, INDEX_FILE)

    if not os.path.isfile(config_path) or not os.path.isfile(index_path):
        raise FileNotFoundError(
            "源目录必须包含 config.json 和 model.safetensors.index.json"
        )

    config = read_json(config_path)
    index = read_json(index_path)

    language_config, language_config_path = (
        resolve_language_config(config)
    )

    if "weight_map" not in index:
        raise KeyError(
            "model.safetensors.index.json 缺少 weight_map"
        )

    weight_map = repair_weight_map_from_shards(
        src,
        index["weight_map"],
    )
    total = int(
        language_config["num_hidden_layers"]
    )

    print(
        f"识别到语言模型配置位置："
        f"{language_config_path}"
    )

    # 先识别权重中的层命名方式，再进行任何层号分析。
    LAYER_PREFIX = detect_layer_prefix(weight_map, total)

    dense_count = int(
        language_config.get("first_k_dense_replace", 0) or 0
    )
    hash_count = int(
        language_config.get("num_hash_layers", 0) or 0
    )

    protected_prefix_count, protected_prefix_details = (
        infer_protected_prefix_count(
            language_config=language_config,
            total=total,
            dense_count=dense_count,
            hash_count=hash_count,
        )
    )

    nextn_count = int(
        language_config.get("num_nextn_predict_layers", 0) or 0
    )

    minimax_m3 = is_minimax_m3_model(config, language_config)

    # MiniMax-M3 的 MTP 权重是独立的 model.mtp.* / mtp.* 结构，
    # num_nextn_predict_layers 不代表 language_model.model.layers.<N> 的尾部层。
    # DeepSeek 等原有模型继续沿用原逻辑。
    tail_count = 0 if minimax_m3 else nextn_count
    configured_extra_layer_count = 0 if minimax_m3 else nextn_count

    # 默认主模型向上取整切半：43 -> 22，61 -> 31。
    target = (
        requested_layers
        if requested_layers is not None
        else (total + 1) // 2
    )

    quant_kind, quant_config_path = detect_quant_kind(
        config, language_config
    )
    check_auto_map(src, config, language_config)

    (
        keep_main,
        protected_prefix_ids,
        sampled_middle,
        tail_ids,
    ) = choose_layers(
        total=total,
        target=target,
        protected_prefix_count=protected_prefix_count,
        tail_count=tail_count,
    )

    # DeepSeek 等模型可能把 NextN/MTP 作为主层编号空间之外的额外层。
    # MiniMax-M3 不走该逻辑，它的独立 MTP tensor 会在下面原名保留。
    keep_extra, ignored_extra = choose_extra_layers(
        weight_map=weight_map,
        total=total,
        configured_count=configured_extra_layer_count,
    )

    # 形如 mtp.* / model.mtp.* / nextn.* 的独立权重。
    standalone_nextn = find_standalone_nextn_tensors(weight_map)

    if standalone_nextn:
        print(
            f"检测到独立 NextN/MTP tensor：{len(standalone_nextn)} 个"
        )
        for name in standalone_nextn[:10]:
            print(" ", name)
        if len(standalone_nextn) > 10:
            print("  ...")

    # MiniMax-M3 的公开 checkpoint 可能在 config 中声明
    # num_nextn_predict_layers / num_mtp_modules，但主 checkpoint 本身并不一定
    # 携带独立 MTP tensor。对于 MiniMax-M3，这些字段属于模型配置元数据，
    # 不应仅因为当前 weight_map 中没有 mtp.* 就擅自改成 0。
    #
    # DeepSeek 等旧布局仍维持原逻辑：只有确实存在额外层或独立 MTP tensor
    # 时才保留 num_nextn_predict_layers。
    has_nextn_weights = bool(keep_extra or standalone_nextn)

    if minimax_m3:
        output_nextn_count = nextn_count
        if nextn_count and not has_nextn_weights:
            print(
                "提示：MiniMax-M3 config 声明 num_nextn_predict_layers="
                f"{nextn_count}，但当前 checkpoint 未发现独立 MTP tensor；"
                "将保留该配置值，但不会把它解释为主模型额外层。"
            )
    else:
        output_nextn_count = nextn_count if has_nextn_weights else 0
        if nextn_count and not has_nextn_weights:
            print(
                "警告：config 声明了 NextN/MTP，"
                "但 checkpoint 中没有找到对应额外层或独立 MTP tensor；"
                "输出 num_nextn_predict_layers 将设置为 0。"
            )

    (
        name_map,
        main_mapping,
        extra_mapping,
    ) = build_name_map(
        weight_map=weight_map,
        total=total,
        keep_main=keep_main,
        keep_extra=keep_extra,
    )

    check_scale_pairs(weight_map, name_map)

    print("\n层选择结果：")
    print(
        f"外层模型类型：{config.get('model_type')}"
    )
    print(
        f"语言模型类型："
        f"{language_config.get('model_type')}"
    )
    print(f"源主模型层数：{total}")
    print(f"目标主模型层数：{target}")
    print(f"first_k_dense_replace：{dense_count}")
    print(f"num_hash_layers：{hash_count}")
    print(f"MiniMax-M3 模式：{minimax_m3}")
    print(f"前置结构推断：{protected_prefix_details}")
    print(f"保留前置结构层：{protected_prefix_ids}")
    print(f"保留主模型末尾层：{tail_ids}")
    print(f"中间采样层：{sampled_middle}")
    print(f"最终保留主模型层：{keep_main}")
    print(f"额外层：{keep_extra}")
    print(f"额外层重编号：{extra_mapping}")
    print(f"输出 NextN/MTP 数量：{output_nextn_count}")

    if ignored_extra:
        print(f"未保留的其他额外层：{ignored_extra}")

    selected_by_shard = defaultdict(list)

    for old_name in name_map:
        selected_by_shard[weight_map[old_name]].append(old_name)

    for names in selected_by_shard.values():
        names.sort()

    writer = ShardWriter(dst, shard_size)
    dtype_counts = defaultdict(int)

    for filename, old_names in tqdm(
        sorted(selected_by_shard.items()),
        desc="processing shards",
    ):
        path = os.path.join(src, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"index 引用了不存在的文件：{path}"
            )

        with safe_open(path, framework="pt", device="cpu") as f:
            missing = set(old_names) - set(f.keys())

            if missing:
                raise ValueError(
                    f"{filename} 缺少 tensor：{sorted(missing)[:20]}"
                )

            for old_name in old_names:
                tensor = f.get_tensor(old_name)
                dtype_counts[str(tensor.dtype)] += 1
                writer.add(name_map[old_name], tensor)
                del tensor

        gc.collect()

    new_weight_map, total_size = writer.finish()

    if quant_kind == "w8a8":
        int8_count = (
            dtype_counts.get("torch.int8", 0)
            + dtype_counts.get("torch.uint8", 0)
        )

        if int8_count == 0:
            raise ValueError(
                "配置显示为 W8A8，但输出中未发现 INT8 权重。"
            )

    elif quant_kind == "fp8":
        fp8_count = sum(
            count
            for dtype_name, count in dtype_counts.items()
            if "float8" in dtype_name.lower()
        )

        if fp8_count == 0:
            raise ValueError(
                "配置显示为 FP8，但输出中未发现 float8 权重。"
                "请检查当前 PyTorch/safetensors 是否支持源模型的 FP8 dtype。"
            )

    copy_aux_files(src, dst)

    source_quant_core = quant_core_snapshot(
        config, language_config
    )

    new_config = copy.deepcopy(config)
    remove_name_or_path(new_config)

    new_language_config, new_config_path = (
        resolve_language_config(
            new_config
        )
    )

    new_language_config[
        "num_hidden_layers"
    ] = target

    new_language_config[
        "num_nextn_predict_layers"
    ] = output_nextn_count

    if (
        "first_k_dense_replace"
        in new_language_config
    ):
        new_language_config[
            "first_k_dense_replace"
        ] = min(
            dense_count,
            target,
        )

    if (
        "num_hash_layers"
        in new_language_config
    ):
        new_language_config[
            "num_hash_layers"
        ] = min(
            hash_count,
            target,
        )

    per_layer_changes = update_per_layer_config(
        config=new_language_config,
        keep_main=keep_main,
        total_main=total,
        source_nextn_count=nextn_count,
        output_nextn_count=output_nextn_count,
    )

    full_layer_mapping = dict(main_mapping)
    full_layer_mapping.update(extra_mapping)

    modules_to_not_convert_changes = remap_modules_to_not_convert(
        outer_config=new_config,
        language_config=new_language_config,
        layer_mapping=full_layer_mapping,
        total_source_layers=total,
    )

    target_quant_core = quant_core_snapshot(
        new_config, new_language_config
    )
    if target_quant_core != source_quant_core:
        raise RuntimeError(
            "量化核心配置被意外修改；除 "
            "modules_to_not_convert 的层号重映射外，"
            "quantization/compression 配置必须保持不变。"
        )

    write_json(os.path.join(dst, CONFIG_FILE), new_config)

    metadata = dict(index.get("metadata", {}))
    metadata["total_size"] = total_size

    write_json(
        os.path.join(dst, INDEX_FILE),
        {
            "metadata": metadata,
            "weight_map": new_weight_map,
        },
    )

    write_json(
        os.path.join(dst, "layer_mapping.json"),
        {
            "outer_model_type": config.get("model_type"),
            "language_model_type": language_config.get("model_type"),
            "language_config_path": language_config_path,
            "source_layer_prefix": LAYER_PREFIX,
            "source_num_hidden_layers": total,
            "target_num_hidden_layers": target,
            "source_first_k_dense_replace": dense_count,
            "source_num_hash_layers": hash_count,
            "source_num_nextn_predict_layers": nextn_count,
            "target_num_nextn_predict_layers": output_nextn_count,
            "is_minimax_m3": minimax_m3,
            "quant_kind": quant_kind,
            "quant_config_path": quant_config_path,
            "protected_prefix_details": protected_prefix_details,
            "configured_extra_layer_count": configured_extra_layer_count,
            "protected_prefix_layers": protected_prefix_ids,
            "sampled_middle_layers": sampled_middle,
            "reserved_main_tail_layers": tail_ids,
            "kept_source_main_layers": keep_main,
            "main_layer_mapping": {
                str(old_id): new_id
                for old_id, new_id in main_mapping.items()
            },
            "kept_source_extra_layers": keep_extra,
            "extra_layer_mapping": {
                str(old_id): new_id
                for old_id, new_id in extra_mapping.items()
            },
            "standalone_nextn_tensor_count": len(standalone_nextn),
            "standalone_nextn_tensor_examples": standalone_nextn[:30],
            "ignored_source_extra_layers": ignored_extra,
            "per_layer_config_changes": per_layer_changes,
            "modules_to_not_convert_changes": modules_to_not_convert_changes,
            "dtype_counts": dict(dtype_counts),
            "total_size": total_size,
        },
    )

    if not skip_validation:
        validate_output(
            dst=dst,
            target_main=target,
            output_extra_layer_count=len(keep_extra),
            output_nextn_count=output_nextn_count,
        )

    print("\n切分完成：", dst)
    print(f"主模型层数：{total} -> {target}")
    print(f"额外层数量：{len(keep_extra)}")
    print(f"输出 NextN/MTP 数量：{output_nextn_count}")
    print(
        "输出语言配置："
        f"{language_config_path}."
        f"num_hidden_layers={target}, "
        f"{language_config_path}."
        f"num_nextn_predict_layers="
        f"{output_nextn_count}"
    )
    print(f"tensor 数：{len(new_weight_map)}")
    print(f"总大小：{total_size / 1024**3:.2f} GB")
    print("dtype 统计：", dict(dtype_counts))

def positive_float(value):
    """argparse 类型：要求参数为正浮点数。"""
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是数字：{value}") from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "裁剪 DeepSeek/GLM/Kimi/MiniMax-M3 等模型的语言层，并保留多模态权重"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  # 默认将主模型层数裁到 ceil(num_hidden_layers / 2)\n"
            "  python prune_moe_layers.py --src /path/to/model --dst /path/to/model-pruned\n\n"
            "  # 指定目标主模型层数和输出 shard 大小\n"
            "  python prune_moe_layers.py --src /path/to/model --dst /path/to/model-32l "
            "--layers 32 --shard-size 4\n\n"
            "注意：源目录必须包含 config.json 和 model.safetensors.index.json。"
        ),
    )

    parser.add_argument("--src", required=True, help="源模型目录")
    parser.add_argument(
        "--dst",
        required=True,
        help="输出模型目录（不能与源目录相同或互相包含）",
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help=(
            "目标主模型层数；不指定时自动使用 "
            "ceil(语言配置 num_hidden_layers / 2)。视觉层和额外层不计入该数量"
        ),
    )

    parser.add_argument(
        "--shard-size",
        type=positive_float,
        default=10.0,
        metavar="GB",
        help="单个输出 shard 最大大小（GB），默认 10；内存较小时建议设为 4 或更小",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录非空时覆盖已有内容",
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过最终输出校验（通常不建议）",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    args = parser.parse_args()

    prune(
        src=args.src,
        dst=args.dst,
        requested_layers=args.layers,
        shard_size=int(args.shard_size * 1024**3),
        overwrite=args.overwrite,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()