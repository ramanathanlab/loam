"""Campaign-local checkpoint policy for Jnana's JSON-backed memory.

Autosaves are update-triggered and at least five seconds apart. Explicit
generation/tournament boundary saves always flush synchronously. This changes
checkpoint timing and JSON whitespace, not the memory schema or ranking math.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from types import MethodType
from typing import Any

log = logging.getLogger(__name__)
FIELDS = ('experiments', 'analyses', 'papers', 'metadata', 'agent_states',
          'datasets', 'training_plans', 'evaluations', 'tournament_state')


def _save(memory: Any, *, reason: str = 'explicit') -> None:
    if not memory.storage_path:
        return
    path = Path(memory.storage_path)
    temporary = None
    started = time.time()
    begun = time.perf_counter()
    # Retain Jnana's reentrant memory lock throughout serialization and replace.
    # No second lock order or background writer can overwrite a newer phase.
    with memory._lock:
        data = {'hypotheses': [h.to_dict() for h in memory.hypotheses.values()],
                **{name: getattr(memory, name) for name in FIELDS}}
        # Freeze the list header also used by the small diagnostic counters.
        data['tournament_state'] = dict(data['tournament_state'])
        if 'matches' in data['tournament_state']:
            data['tournament_state']['matches'] = list(data['tournament_state']['matches'])
        encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
            temporary = Path(name)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        memory._loam_last_checkpoint = time.monotonic()
        memory._loam_checkpoint_sequence += 1
        record = {'sequence': memory._loam_checkpoint_sequence, 'reason': reason,
                  'started_at': started, 'finished_at': time.time(),
                  'seconds': time.perf_counter() - begun, 'json_bytes': len(encoded),
                  'hypotheses': len(data['hypotheses']),
                  'matches': len(data['tournament_state'].get('matches', []))}
    try:
        with (path.parent/'jnana-checkpoint-timings.jsonl').open('a') as stream:
            stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    except OSError:
        log.warning('Checkpoint persisted; optional timing record could not be written')


def _autosave(memory: Any) -> None:
    if not memory.storage_path:
        return
    with memory._lock:
        if time.monotonic() - memory._loam_last_checkpoint >= memory._loam_checkpoint_interval:
            memory.save(reason='autosave')


def install_checkpoint_policy(memory: Any, *, interval_seconds: float = 5.0) -> bool:
    """Install on one owned memory instance; never patch the installed package.

    There is no background timer: deferred updates flush on a later update or
    explicit phase boundary. A crash may therefore lose recent unsaved updates.
    IO failures propagate and leave the previous complete checkpoint intact.
    """
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError('Checkpoint interval must be positive and finite')
    if not memory.storage_path:
        return False
    if hasattr(memory, '_loam_checkpoint_interval'):
        return True
    memory._loam_checkpoint_interval = interval_seconds
    memory._loam_last_checkpoint = float('-inf')
    memory._loam_checkpoint_sequence = 0
    memory.save = MethodType(_save, memory)
    memory._save_if_needed = MethodType(_autosave, memory)
    return True


def checkpointed_coscientist(base: type, *, interval_seconds: float = 5.0) -> type:
    """Preserve the existing scientist/retry class and its explicit stop/save."""
    class CheckpointedScientist(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_checkpoint_policy(self.memory, interval_seconds=interval_seconds)
    return CheckpointedScientist
