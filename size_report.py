#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读 safetensors header（8 字节长度 + JSON），按命名空间/层统计字节数。"""
import json
import os
import struct
import sys
from collections import defaultdict

B = sys.argv[1] if len(sys.argv) > 1 else "/workspace_luwu/DeepSeek-V4.1-Flash-1017-pruned"
GB = 1024 ** 3

hits = defaultdict(int)     # 命名空间 -> bytes
layers = defaultdict(int)   # 层号 -> bytes
big = []                    # 最大的张量

for f in sorted(os.listdir(B)):
    if not f.endswith(".safetensors"):
        continue
    path = os.path.join(B, f)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for name, info in header.items():
        if name == "__metadata__":
            continue
        d0, d1 = info["data_offsets"]
        size = d1 - d0
        parts = name.split(".")
        ns = parts[0]
        hits[ns] += size
        if ns == "layers":
            layers[int(parts[1])] += size
        big.append((size, name))

total = sum(hits.values())
print(f"目录: {B}")
print(f"总大小: {total / GB:.2f} GB")
print("\n== 按命名空间 ==")
for ns, v in sorted(hits.items(), key=lambda kv: -kv[1]):
    print(f"  {ns:12s} {v/GB:9.2f} GB  ({v/total*100:5.1f}%)")

print("\n== 语言层逐层大小 (layers.*) ==")
for lid, v in sorted(layers.items()):
    print(f"  layer {lid:3d}  {v/GB:7.2f} GB")
print(f"  layers 合计 {sum(layers.values())/GB:.2f} GB  平均 {sum(layers.values())/len(layers)/GB:.2f} GB/层")

print("\n== 最大的 8 个张量 ==")
for size, name in sorted(big, reverse=True)[:8]:
    print(f"  {size/GB:8.2f} GB  {name}")