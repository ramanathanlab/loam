#!/usr/bin/env python3
"""Print a non-submitting starting recipe for a node/GPU budget."""
import argparse
import json
from loam.scaling import dense_recipe

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes', type=int, required=True)
    p.add_argument('--gpus-per-node', type=int, default=4)
    p.add_argument('--inference', type=int)
    p.add_argument('--science-workers-per-gpu', type=int, default=2)
    a = p.parse_args()
    recipe = dense_recipe(a.nodes, a.gpus_per_node, inference=a.inference,
                          science_workers_per_gpu=a.science_workers_per_gpu)
    print(json.dumps({'recipe': recipe.as_dict(),
                      'scope': 'Measured density at 20 and 256 Jupiter nodes with four GPUs/node. '
                               'Other layouts require scientific and hardware qualification.'}, indent=2))
