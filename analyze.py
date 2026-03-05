"""
Parses OpenVINO IR XML files and compares operation types between models.
Also extracts layer name mappings (PyTorch module name → OV IR node name).

Usage:
    python analyze.py              # op breakdown + comparison
    python analyze.py --names      # also dump all node names (for NNCF mapping)
"""
import xml.etree.ElementTree as ET
import os
import sys
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


def get_layer_names(xml_path):
    """Extract all layer names from the IR XML, grouped by op type."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = []
    for layer in root.iter("layer"):
        name = layer.get("name", "")
        op_type = layer.get("type", "")
        layers.append((name, op_type))
    return layers


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


def dump_names(xml_path, filter_types=None):
    """Print all node names, optionally filtered by op type (e.g. MatMul)."""
    print(f"\n{'='*60}")
    print(f"Layer names: {xml_path}")
    if filter_types:
        print(f"Filtered to: {filter_types}")
    print(f"{'='*60}")
    layers = get_layer_names(xml_path)
    for name, op_type in layers:
        if filter_types and op_type not in filter_types:
            continue
        print(f"  [{op_type:20s}]  {name}")
    return layers


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

show_names = "--names" in sys.argv

all_ops = {}
for path in xml_files:
    name = os.path.basename(path).replace(".xml", "")
    all_ops[name] = analyze(path)
    if show_names:
        # Show MatMul and Convolution nodes — these are the weight-bearing ops
        # that NNCF's ignored_scope targets
        dump_names(path, filter_types={"MatMul", "Convolution"})

# If we have multiple models, compare them
keys = list(all_ops.keys())
if len(keys) >= 2:
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            compare(all_ops[keys[i]], keys[i], all_ops[keys[j]], keys[j])
