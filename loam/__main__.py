"""Run a campaign from its sole user-authored YAML configuration."""

import argparse
import asyncio
import logging

from .config import load
from .launcher import managed_hpc_server
from .orchestrator import Campaign


async def main(config_path: str) -> None:
    cfg = load(config_path)
    async with managed_hpc_server(cfg):
        await Campaign(cfg).run(cfg.rounds)


parser = argparse.ArgumentParser(prog='loam')
parser.add_argument('config')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
asyncio.run(main(args.config))
