#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复校 DeepSeek-V4.1-Flash 裁剪产物（不依赖 prune 脚本自身的校验）。

只做只读检查：
  1) 目录/索引/分片结构
  2) 期望张量名集合 == 实际输出集合（逐名字推导，不信任 index）
  3) 每个张量的 dtype/shape 与源 checkpoint 对应张量一致
  4) 逐层张量数量一致、专家数量一致
  5) config 层面：层数、逐层数组、跨层引用、engram、vision、量化配置
  6) 抽样做真实数据比对（torch.equal）
"""
import json
import os
import sys
import random

import torch
from safetensors import safe_open

SRC = "/mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V4.1-Flash-1017"
DST = "/workspace_luwu/DeepSeek-V4.1-Flash-1017-pruned"

fails = []
warns = []


def ok(msg):
    print("  PASS  " + msg)


def bad(msg):
    print("  FAIL  " + msg)
    fails.append(msg)


def warn(msg):
    print("  WARN  " + msg)
    warns.append(msg)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- 0. 基础结构 ----------------
print("== 0. 目录与索引 ==")
src_idx = load(os.path.join(SRC, "model.safetensors.index.json"))
dst_idx = load(os.path.join(DST, "model.safetensors.index.json"))
src_wm = src_idx["weight_map"]
dst_wm = dst_idx["weight_map"]
src_cfg = load(os.path.join(SRC, "config.json"))
dst_cfg = load(os.path.join(DST, "config.json"))

dst_shards = sorted(set(dst_wm.values()))
missing_files = [s for s in dst_shards if not os.path.isfile(os.path.join(DST, s))]
ok(f"输出张量数 {len(dst_wm)}，分片 {len(dst_shards)} 个")
if missing_files:
    bad(f"index 引用了不存在的分片: {missing_files}")
else:
    ok("index 引用的分片全部存在")

orphan = sorted(f for f in os.listdir(DST)
                if f.endswith(".safetensors") and f not in dst_shards)
if orphan:
    bad(f"目录里有 index 未登记的分片: {orphan}")
else:
    ok("目录中没有游离的 .safetensors 文件")

# 打开句柄
src_h = {}
for s in sorted(set(src_wm.values())):
    src_h[s] = safe_open(os.path.join(SRC, s), framework="pt", device="cpu")
dst_h = {}
for s in dst_shards:
    dst_h[s] = safe_open(os.path.join(DST, s), framework="pt", device="cpu")

# ---------------- 1. 分片内 key 集合 ----------------
print("== 1. 分片 key 集合 vs index ==")
by_shard = {}
for name, shard in dst_wm.items():
    by_shard.setdefault(shard, set()).add(name)
dup = {}
all_names = set()
for name in dst_wm:
    if name in all_names:
        dup[name] = True
    all_names.add(name)
if dup:
    bad(f"index 中重复张量名: {list(dup)[:5]}")
else:
    ok("index 内张量名无重复")

bad_shards = []
for shard, expected in sorted(by_shard.items()):
    actual = set(dst_h[shard].keys())
    if actual != expected:
        bad_shards.append((shard, len(expected - actual), len(actual - expected)))
if bad_shards:
    bad(f"{len(bad_shards)} 个分片的 key 与 index 不一致: {bad_shards[:3]}")
else:
    ok(f"{len(by_shard)} 个分片的 key 与 index 完全一致")

# ---------------- 2. 由源推导期望集合 ----------------
print("== 2. 期望张量名集合（由 layer_mapping 推导）==")
lm = load(os.path.join(DST, "layer_mapping.json"))
mapping = {int(k): int(v) for k, v in lm["main_layer_mapping"].items()}
inv = {v: k for k, v in mapping.items()}
kept = set(mapping)

expected = {}
for name in src_wm:
    if name.startswith("layers."):
        old_id = int(name.split(".")[1])
        if old_id not in kept:
            continue
        new_name = f"layers.{mapping[old_id]}." + name.split(".", 2)[2]
    else:
        new_name = name
    expected[new_name] = name

if set(expected) == set(dst_wm):
    ok(f"输出集合 == 期望集合（{len(expected)} 个张量）")
else:
    miss = sorted(set(expected) - set(dst_wm))[:5]
    extra = sorted(set(dst_wm) - set(expected))[:5]
    bad(f"输出集合与期望不一致: 缺 {len(set(expected) - set(dst_wm))} 个 {miss}, "
        f"多 {len(set(dst_wm) - set(expected))} 个 {extra}")

# ---------------- 3. dtype / shape ----------------
print("== 3. dtype 与 shape 逐个比对 ==")
mismatch = []
for new_name, old_name in expected.items():
    if new_name not in dst_wm or old_name not in src_wm:
        continue
    s_slice = src_h[src_wm[old_name]].get_slice(old_name)
    d_slice = dst_h[dst_wm[new_name]].get_slice(new_name)
    if s_slice.get_shape() != d_slice.get_shape() or s_slice.get_dtype() != d_slice.get_dtype():
        mismatch.append((new_name, old_name,
                         str(s_slice.get_dtype()), s_slice.get_shape(),
                         str(d_slice.get_dtype()), d_slice.get_shape()))
if mismatch:
    bad(f"{len(mismatch)} 个张量 dtype/shape 不一致，示例: {mismatch[:3]}")
else:
    ok(f"{len(expected)} 个张量的 dtype/shape 全部与源一致")

# ---------------- 4. 逐层张量数量 / 专家数 ----------------
print("== 4. 逐层张量数量与专家数量 ==")
src_cnt = {}
for name in src_wm:
    if name.startswith("layers."):
        src_cnt.setdefault(int(name.split(".")[1]), 0)
        src_cnt[int(name.split(".")[1])] += 1
dst_cnt = {}
for name in dst_wm:
    if name.startswith("layers."):
        dst_cnt.setdefault(int(name.split(".")[1]), 0)
        dst_cnt[int(name.split(".")[1])] += 1

cnt_bad = []
for new_id, old_id in inv.items():
    if src_cnt.get(old_id) != dst_cnt.get(new_id):
        cnt_bad.append((old_id, new_id, src_cnt.get(old_id), dst_cnt.get(new_id)))
if cnt_bad:
    bad(f"逐层张量数量不一致: {cnt_bad[:5]}")
else:
    ok(f"{len(inv)} 个保留层逐一对应，张量数量一致")

def expert_ids(prefix, wm):
    ids = set()
    key = ".ffn.experts."
    for name in wm:
        if name.startswith(prefix) and key in name:
            ids.add(int(name.split(key)[1].split(".")[0]))
    return sorted(ids)

exp_bad = []
for new_id, old_id in inv.items():
    a = expert_ids(f"layers.{old_id}.", src_wm)
    b = expert_ids(f"layers.{new_id}.", dst_wm)
    if a != b:
        exp_bad.append((old_id, new_id, len(a), len(b)))
if exp_bad:
    bad(f"专家编号集合不一致: {exp_bad[:3]}")
else:
    ok(f"各层专家集合一致（示例 experts=0..{len(a)-1}，共 {len(a)} 个）")

# ---------------- 5. config ----------------
print("== 5. config 一致性 ==")
sc, dc = src_cfg["text_config"], dst_cfg["text_config"]
if dc["num_hidden_layers"] == 20:
    ok("num_hidden_layers = 20")
else:
    bad(f"num_hidden_layers = {dc['num_hidden_layers']}")

if dc["num_nextn_predict_layers"] == sc["num_nextn_predict_layers"] == 3:
    ok("num_nextn_predict_layers = 3（与源一致）")
else:
    bad(f"num_nextn_predict_layers = {dc['num_nextn_predict_layers']}")

cr = dc["compress_ratios"]
if len(cr) == dc["num_hidden_layers"] + dc["num_nextn_predict_layers"]:
    ok(f"compress_ratios 长度 {len(cr)} = 20 + 3")
else:
    bad(f"compress_ratios 长度 {len(cr)} != 23")
if cr[:20] == [sc["compress_ratios"][o] for o in sorted(kept)]:
    ok(f"compress_ratios 主层采样正确: {cr[:20]}")
else:
    bad("compress_ratios 主层采样与保留层不匹配")

N = dc["num_hidden_layers"]
for field in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids",
              "dspark_target_layer_ids"):
    vals = dc[field]
    if all(isinstance(v, int) and 0 <= v < N for v in vals) and vals == sorted(set(vals)):
        ok(f"{field} = {vals}（均在 0..{N-1} 且严格递增）")
    else:
        bad(f"{field} = {vals} 越界或未排序")

if dc["candidate_source_layer_id"] < N and dc["candidate_source_layer_id"] >= 0:
    ok(f"candidate_source_layer_id = {dc['candidate_source_layer_id']}")
else:
    bad(f"candidate_source_layer_id = {dc['candidate_source_layer_id']}")

if len(dc["engram_num_embeddings"]) == len(dc["engram_layer_ids"]):
    ok(f"engram_num_embeddings 与 engram_layer_ids 对齐: {dc['engram_num_embeddings']}")
else:
    bad("engram_num_embeddings 与 engram_layer_ids 长度不一致")

for lid in dc["engram_layer_ids"]:
    has = [n for n in dst_wm if n.startswith(f"layers.{lid}.engram.")]
    if has:
        ok(f"engram 层 {lid} 权重存在（{len(has)} 个张量）")
    else:
        bad(f"engram 层 {lid} 缺少 engram 权重")

for lid in dc["kv_source_layer_ids"]:
    has = [n for n in dst_wm if n.startswith(f"layers.{lid}.attn.compressor.")]
    if has:
        ok(f"kv_source 层 {lid} 有 compressor 权重")
    else:
        bad(f"kv_source 层 {lid} 缺 compressor 权重")

for lid in dc["index_source_layer_ids"]:
    has = [n for n in dst_wm if n.startswith(f"layers.{lid}.attn.indexer.")]
    if has:
        ok(f"index_source 层 {lid} 有 indexer 权重")
    else:
        bad(f"index_source 层 {lid} 缺 indexer 权重")

# dspark 目标层 == MTP 对应的主干最后三层
if dc["dspark_target_layer_ids"] == list(range(N - 3, N)):
    ok(f"dspark_target_layer_ids 指向主干最后三层 {dc['dspark_target_layer_ids']}")
else:
    bad(f"dspark_target_layer_ids = {dc['dspark_target_layer_ids']}，应为 {list(range(N-3, N))}")

if dst_cfg["vision_config"] == src_cfg["vision_config"]:
    ok("vision_config 与源完全一致（视觉塔未裁）")
else:
    bad("vision_config 被改动")

if dst_cfg["quantization_config"] == src_cfg["quantization_config"]:
    ok(f"quantization_config 与源一致: {dst_cfg['quantization_config']}")
else:
    bad("quantization_config 与源不一致")

vision_n = len([n for n in dst_wm if n.startswith("vision.")])
src_vision_n = len([n for n in src_wm if n.startswith("vision.")])
if vision_n == src_vision_n:
    ok(f"视觉塔张量 {vision_n} 个，与源一致")
else:
    bad(f"视觉塔张量 {vision_n} != 源 {src_vision_n}")

mtp_n = len([n for n in dst_wm if n.startswith("mtp.")])
src_mtp_n = len([n for n in src_wm if n.startswith("mtp.")])
if mtp_n == src_mtp_n:
    ok(f"MTP 张量 {mtp_n} 个，与源一致")
else:
    bad(f"MTP 张量 {mtp_n} != 源 {src_mtp_n}")

# scale / weight 配对
print("== 6. 量化 weight/scale 配对 ==")
names = set(dst_wm)
pair_bad = []
for name in names:
    if name.endswith(".scale"):
        base = name[:-len(".scale")]
        if base + ".weight" not in names and ".experts." in name:
            pair_bad.append(name)
for name in names:
    if name.endswith(".weight") and ".experts." in name:
        if name[:-len(".weight")] + ".scale" not in names:
            pair_bad.append(name)
if pair_bad:
    bad(f"{len(pair_bad)} 个专家权重/scale 未配对: {pair_bad[:5]}")
else:
    ok("专家权重与 scale 全部成对")

# ---------------- 7. 真实数据抽样比对 ----------------
print("== 7. 抽样数据比对（torch.equal）==")
per_shard = {}
for name, shard in dst_wm.items():
    per_shard.setdefault(shard, []).append(name)

samples = []
for shard, ns in sorted(per_shard.items()):
    samples.append(random.choice(sorted(ns)))
extra = [n for n in (
    "layers.0.attn.wq_a.weight",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.engram.embed.weight",
    "mtp.0.attn.wkv.weight",
    "vision.patch_embed.proj.weight",
    "head.weight",
    "embed.weight",
    "norm.weight",
) if n in dst_wm]
samples = list(dict.fromkeys(samples + extra))

same = 0
diff = []
for new_name in samples:
    old_name = expected[new_name]
    a = src_h[src_wm[old_name]].get_tensor(old_name)
    b = dst_h[dst_wm[new_name]].get_tensor(new_name)
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        diff.append(new_name)
    else:
        same += 1
if diff:
    bad(f"{len(diff)} 个抽样张量数据不一致: {diff[:5]}")
else:
    ok(f"{same} 个抽样张量逐元素与源一致")

# ---------------- 汇总 ----------------
print()
print("=" * 60)
print(f"FAIL={len(fails)}  WARN={len(warns)}")
for f in fails:
    print("  FAIL:", f)
print("RESULT:", "PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)