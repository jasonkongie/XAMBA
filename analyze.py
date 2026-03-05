"""
Parses OpenVINO IR XML files and compares operation types between models.
Usage: python analyze.py
"""
import xml.etree.ElementTree as ET
import os
from collections import Counter


def get_ops(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ops = Counter()
    for layer in root.iter("layer"):
        op_type = layer.get("type")
        if op_type:
            ops[op_type] += 1
    return ops


def analyze(xml_path):
    print(f"\n{'='*60}")
    print(f"Model: {xml_path}")
    print(f"{'='*60}")
    ops = get_ops(xml_path)
    print(f"Total ops: {sum(ops.values())}  |  Unique op types: {len(ops)}")
    print("\nOp breakdown (count  type):")
    for op, count in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {op}")
    return ops


def compare(ops1, name1, ops2, name2):
    print(f"\n{'='*60}")
    print(f"Diff: ops in [{name1}] but NOT in [{name2}]")
    print(f"{'='*60}")
    only_in_1 = set(ops1) - set(ops2)
    if only_in_1:
        for op in sorted(only_in_1):
            print(f"  {ops1[op]:4d}  {op}")
    else:
        print("  (none)")

    print(f"\nDiff: ops in [{name2}] but NOT in [{name1}]")
    print(f"{'='*60}")
    only_in_2 = set(ops2) - set(ops1)
    if only_in_2:
        for op in sorted(only_in_2):
            print(f"  {ops2[op]:4d}  {op}")
    else:
        print("  (none)")


# Find all XML files in ov_model/
xml_files = sorted([
    os.path.join("ov_model", f)
    for f in os.listdir("ov_model")
    if f.endswith(".xml")
])

if not xml_files:
    print("No XML files found in ov_model/")
    exit(1)

all_ops = {}
for path in xml_files:
    name = os.path.basename(path).replace(".xml", "")
    all_ops[name] = analyze(path)

# If we have both mamba1 and mamba2, compare them
keys = list(all_ops.keys())
if len(keys) >= 2:
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            compare(all_ops[keys[i]], keys[i], all_ops[keys[j]], keys[j])
