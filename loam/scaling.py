"""The scaling recipe: campaign settings sized to an allocation, from measured workload density.

Moeen measured one dense recipe on Jupiter, at 20 and 256 nodes of four GH200 GPUs
and 288 CPU cores each (docs/scaling-recipe-book.md). The density formula scales
to any node count and GPU count, but the numbers are only measured on that System
and node shape, so `recipe` refuses any other unless the caller accepts an
unmeasured recipe, and then warns.

    loam-scaling --system jupiter --nodes 20
    loam-scaling --config loam.yaml

prints the recipe, whether it is measured, where it was measured, and its
campaign config settings. It never stages or submits anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, fields

from .config import load
from .systems import SYSTEMS, System


@dataclass(frozen=True)
class Measurement:
    """Where a recipe's numbers were measured: a System, its node shape, and the node counts run."""
    system: str
    gpus_per_node: int
    cpus_per_node: int
    node_counts: tuple[int, ...]
    science_workers_per_gpu: int
    source: str


# Jupiter jobs 2979745 (20 nodes, "20i") and 2999784 (256 nodes, "256c"), November 2026.
JUPITER_DENSE = Measurement(system='jupiter', gpus_per_node=4, cpus_per_node=288, node_counts=(20, 256),
                            science_workers_per_gpu=2, source='docs/scaling-recipe-book.md')


class UnmeasuredScaling(ValueError):
    """A recipe was asked for on a System or node shape no measurement ran on."""


class UnmeasuredScalingWarning(UserWarning):
    """A recipe was given for a configuration no measurement ran: plan on it as a guess, then qualify it."""


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
    # The measurement this density comes from, and whether this exact configuration is one it ran.
    basis: Measurement
    measured: bool
    generation_batch_size: int = 4
    swiss_rounds: int = 1
    rounds: int = 3

    def campaign_settings(self) -> dict:
        """This recipe as campaign config values, nested as the campaign config nests them.

        The inference : science split has no campaign config key; single-allocation staging places it.
        """
        return {
            'rounds': self.rounds,
            'concurrency': self.concurrency,
            'synthesis_concurrency': self.synthesis_concurrency,
            'tournament': {
                'top_k': self.top_k,
                'initial_hypotheses': self.initial_hypotheses,
                'new_per_round': self.new_per_round,
                'gen_batch_size': self.generation_batch_size,
                'swiss_rounds': self.swiss_rounds,
                'workers': self.tournament_workers,
            },
            'execution': {'hpc': {'science_workers_per_gpu': self.science_workers_per_gpu}},
        }


def recipe(system: System, nodes: int, *, inference: int | None = None,
           science_workers_per_gpu: int | None = None, allow_unmeasured: bool = False) -> ScalingRecipe:
    """Moeen's dense recipe for `nodes` nodes of `system` (with its node shape).

    `inference` overrides the inference : science split, and `science_workers_per_gpu` the GPU sharing
    (by default the measured two, or the System's limit if lower). Raises UnmeasuredScaling for a System
    or node shape the recipe was not measured on, unless `allow_unmeasured`. A recipe for anything the
    measurement did not run warns UnmeasuredScalingWarning and says it is not measured.
    """
    if type(nodes) is not int or nodes < 2:
        raise ValueError('A recipe needs at least two nodes: one for inference and one for science')
    basis = JUPITER_DENSE
    default_inference = max(1, min(nodes - 1, round(.4 * nodes)))
    inference = default_inference if inference is None else inference
    if type(inference) is not int or not 0 < inference < nodes:
        raise ValueError(f'inference must leave at least one of the {nodes} nodes for science')
    default_workers = min(basis.science_workers_per_gpu, system.max_science_workers_per_gpu)
    workers = default_workers if science_workers_per_gpu is None else science_workers_per_gpu
    if type(workers) is not int or workers < 1:
        raise ValueError('science_workers_per_gpu must be a positive integer')
    system.check_science_workers_per_gpu(workers)

    gpus, cpus = system.gpus_per_node, len(system.cpu_ids)
    if (system.name, gpus, cpus) != (basis.system, basis.gpus_per_node, basis.cpus_per_node):
        problem = (f'The scaling recipe was measured only on {basis.system} with {basis.gpus_per_node} GPUs and '
                   f'{basis.cpus_per_node} CPUs per node (see {basis.source}); {system.name} with {gpus} GPUs '
                   f'and {cpus} CPUs per node is unmeasured')
        if not allow_unmeasured:
            raise UnmeasuredScaling(problem + '. Pass allow_unmeasured to get an unqualified starting point.')
        unmeasured = [problem]
    else:
        unmeasured = []
        if nodes not in basis.node_counts:
            counts = ' and '.join(str(count) for count in basis.node_counts)
            unmeasured.append(f'it was measured at {counts} nodes, not at {nodes} nodes')
        if inference != default_inference:
            unmeasured.append(f'it was measured with {default_inference} inference nodes of {nodes}, not {inference}')
        if workers != basis.science_workers_per_gpu:
            unmeasured.append(f'it was measured with {basis.science_workers_per_gpu} science workers per GPU, '
                              f'not {workers}')
        if unmeasured:
            unmeasured = [f'The scaling recipe on {basis.system} is unmeasured here: ' + '; '.join(unmeasured)]
    if unmeasured:
        warnings.warn(unmeasured[0] + f'. Qualify a small layout before planning on it (see {basis.source}).',
                      UnmeasuredScalingWarning, stacklevel=2)

    science = nodes - inference
    # C = half the logical science slots; selected backlog = 3C; initial pool = 2 * backlog.
    concurrency = math.ceil(science * gpus * workers / 2)
    top_k = 3 * concurrency
    return ScalingRecipe(nodes=nodes, inference=inference, science=science, gpus_per_node=gpus,
                         science_workers_per_gpu=workers, concurrency=concurrency, top_k=top_k,
                         initial_hypotheses=2 * top_k, new_per_round=math.ceil(top_k / 2),
                         tournament_workers=2 * inference, synthesis_concurrency=inference,
                         basis=basis, measured=not unmeasured)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='loam-scaling', description=__doc__.split('\n\n')[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help="a campaign config; its System, node shape and node count are used")
    source.add_argument('--system', choices=sorted(SYSTEMS))
    parser.add_argument('--nodes', type=int, help="required with --system; overrides the config's")
    parser.add_argument('--gpus-per-node', type=int, help="with --system; default: the System's")
    parser.add_argument('--cpus-per-node', type=int, help="with --system; default: the System's")
    parser.add_argument('--inference', type=int, help='inference nodes; default: the measured split')
    parser.add_argument('--science-workers-per-gpu', type=int, help='default: as measured, up to the System limit')
    parser.add_argument('--allow-unmeasured', action='store_true',
                        help='give an unqualified recipe for an unmeasured System or node shape, with a warning')
    args = parser.parse_args(argv)
    if args.config:
        if args.gpus_per_node or args.cpus_per_node:
            parser.error('--gpus-per-node and --cpus-per-node come from the campaign config with --config')
        hpc = load(args.config).execution.hpc
        if hpc is None:
            parser.error(f'{args.config} has no execution.hpc deployment to size')
        system, nodes = hpc.system_definition(), args.nodes or hpc.allocation.nodes
    else:
        if args.nodes is None:
            parser.error('--nodes is required with --system')
        try:
            system = SYSTEMS[args.system].with_node_shape(gpus=args.gpus_per_node, cpus=args.cpus_per_node)
        except ValueError as error:
            parser.error(str(error))
        nodes = args.nodes
    try:
        result = recipe(system, nodes, inference=args.inference, science_workers_per_gpu=args.science_workers_per_gpu,
                        allow_unmeasured=args.allow_unmeasured)
    except ValueError as error:
        print(f'loam-scaling: {error}', file=sys.stderr)
        sys.exit(1)
    print(json.dumps({
        'recipe': {f.name: getattr(result, f.name) for f in fields(result) if f.name not in ('basis', 'measured')},
        'measured': result.measured,
        'measured_on': asdict(result.basis),
        'campaign_settings': result.campaign_settings(),
    }, indent=2))


if __name__ == '__main__':
    main()
