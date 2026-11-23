"""Saturation accounting: what a campaign's allocation cost and how much reviewed science it bought.

`metrics` turns a staged job into allocated GPU-hours, GPU busy, slot
occupancy, trial throughput and reviewed hypothesis counts, and evaluates the
scale-up gates; it reads allocated time and GPU telemetry through the job's
System (`loam.systems`). `storage` keeps every job directory tied to one
ownership ledger. Run the audit with `loam-saturation` (see `__main__`). `scaleup`
stages a larger successor once a trial passes its gate (`loam-scaleup`).
"""
