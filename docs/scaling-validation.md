# Preparation validation

The following checks were performed on November 20, 2026 before publishing this
branch. No new Slurm campaign was submitted during code preparation.

- A fresh Python 3.12 environment was installed from `uv.lock` with the `dev`,
  `hpc` and `reports` extras. The complete offline suite passed **282 tests**.
- The controller environment used during the campaigns also passed the earlier
  257-test suite; the scientific environment separately passed all 11 compact-
  coordinate provenance regressions. The fresh full suite includes those tests.
- Python sources parsed with Python 3.11 grammar; active shell scripts passed
  `bash -n`. GitHub CI checks Python 3.11 and 3.12 using the locked dependencies.
- Both measured recipe wrappers staged fresh 20/256-node jobs without submitting
  them. Their source/native/input seals, paths, role counts and model profiles
  passed preflight. A mocked submission regression checked one parent `sbatch`,
  site account/partition/resources, receipt persistence and refusal to resubmit.
- All **33 files in the local external scientific input bundle are byte-identical**
  to the final 256c inputs. Only scientific programs and checksums are committed;
  raw structures, annotations, original prompts and proposals remain external. Evaluating the native serving-argument expressions confirmed that the
  four-GPU vLLM argument list equals the measured final campaign's profile.
- CPU regressions cover changed GPU-slot counts, source-tamper detection,
  `top_k` backlog admission, synthesis coverage/context bounds, cache invalidation,
  registry/disk equivalence, checkpoint failure handling, model-setting forwarding,
  MCP initialization and staggered rate-limit recovery.
- Publication checks excluded campaign outputs/environments and found no matching
  owned service credentials, private-key payloads or credential-token patterns.
  Historical operational extracts omit credentials and research context.

The renamed package, fresh-checkout staging and generalized Slurm GPU/CPU
parameters have **not** been run in a new GPU campaign. Historical performance
numbers belong to their original source/configuration hashes. The 256c workflow
completed three rounds but failed claim-provenance qualification and the 50%
whole-allocation GPU-busy target. Code preparation does not change that status.
