"""Measured workload density, expressed independently of a machine's GPU count.

Only 20/256-node, four-GPU Jupiter deployments have been measured. Other
hardware/node counts are qualification candidates, not throughput predictions.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ScalingRecipe:
    nodes: int
    inference: int
    science: int
    gpus_per_node: int
    science_workers_per_gpu: int
    concurrency: int
    top_k: int
    initial_hypotheses: int
    new_per_round: int
    tournament_workers: int
    synthesis_concurrency: int
    generation_batch_size: int = 4
    swiss_rounds: int = 1
    rounds: int = 3

    def as_dict(self):
        return asdict(self)


def dense_recipe(nodes: int, gpus_per_node: int = 4, *, inference: int | None = None,
                 science_workers_per_gpu: int = 2) -> ScalingRecipe:
    """C=half the logical science slots; selected backlog=3C; initial pool=6C."""
    if type(nodes) is not int or nodes < 2:
        raise ValueError('Dedicated inference/science roles require at least two nodes')
    if type(gpus_per_node) is not int or gpus_per_node < 1:
        raise ValueError('gpus_per_node must be a positive integer')
    if type(science_workers_per_gpu) is not int or science_workers_per_gpu not in (1, 2):
        raise ValueError('science_workers_per_gpu must be one or two')
    inference = max(1, min(nodes - 1, round(.4 * nodes))) if inference is None else inference
    if type(inference) is not int or not 0 < inference < nodes:
        raise ValueError('Inference nodes must leave a nonempty science pool')
    science = nodes - inference
    concurrency = math.ceil(science * gpus_per_node * science_workers_per_gpu / 2)
    top_k = 3 * concurrency
    return ScalingRecipe(nodes, inference, science, gpus_per_node, science_workers_per_gpu,
                         concurrency, top_k, 2 * top_k, math.ceil(top_k / 2),
                         2 * inference, inference)
