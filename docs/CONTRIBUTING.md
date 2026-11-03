<!-- SPDX-License-Identifier: MIT -->
# Contribute to Arsenal

Contribute a model folder, engine/runner, kernel or measured result with its evidence and research history. Arsenal keeps failed attempts as well as accepted work.

## Prepare the change

Read the affected model's README, pins, settings/ops, gate, existing results and `log/` before choosing an optimization. Use `ADDING-A-MODEL.md` for a new HF model, `RUNNERS.md` for any executable, and `MODEL-FOLDER.md` for declarations. Keep the change inspectable; preserve unrelated work.

- **Model:** immutable source/asset pins, reviewed runner, correct licence, valid folder, Get/Load/ops/unload proof and explicit unsupported capabilities.
- **Runner/engine:** a native engine using the common wire, no idle polling or hidden downloads; include the conformance checker, numerical gate, dependency locks/artifact packaging and memory/cancellation/exit checks.
- **Kernel:** source in the model folder, manifest for chip family × mode, exact reference comparison or Fast quality gate, fallback/unsupported behavior and actual tested chips/dtypes. An entry point compiling is not numerical proof.
- **Results:** reproducible commands, runner/worker hashes, suite/version/hash/profile, actual chip/OS/dtype/mode, gate evidence and measurement conditions. Compare only compatible runs. Quick/stub/contended runs do not qualify for ranking or Measured.

New Arsenal code is MIT (`SPDX-License-Identifier: MIT`). Preserve upstream third-party notices, record copied source path/revision, and list new dependency licences. A model's weight licence remains its vendor's licence. Never include secrets, raw private sessions, weight blobs or unrelated marketing/QA artifacts in a public repository submission.

## Verify and record

```sh
scripts/mlock work runner-unit-tests -- scripts/test.sh --filter RunnerCheckerTests
Arsenal catalog validate models/<id>
Arsenal runner-check <runner argv…> --model models/<id> --weights <verified-snapshot>
```

Use Swift Testing, isolated `ARSENAL_SUPPORT_DIR`/`ARSENAL_CACHE_DIR`, and the model's gate for the actual implementation. On the shared Mac, all builds/heavy CPU use `mlock work`; real GPU proofs require current authorization, `gpulock`, local copies and ≤180 seconds. Controlled benchmarks require a separately authorized quiet window; do not silently turn a smoke into a measurement session. These commands are examples of check types, not proof they ran.

Keep experiments, historical attempts and raw captures under ignored `lab/`; promote only current product source, tests, recipes and compact evidence. Write `log/NNN-slug.md` for a shipped attempt: hypothesis, change, actual measurement, kept/rejected/inconclusive verdict, numbers and evidence links. Keep earlier entries and compatible baselines. Result files must validate against `Resources/schemas/results.schema.json`; log headers against the research-log validator. State unavailable checks as gaps, including quality metrics that lack a qualified reference. Keep evidence outside `/tmp` before handing off; leave no worker processes running.

## Acceptance and release

A submission names the problem, owned files, source revisions, tested configurations, commands/evidence, speed comparison, gate results and known limits. Review checks reproducibility, numerical claims, error paths, idle/process cleanup, trust/licensing and packaging, not just compilation. A fresh independent engineering review checks substantive changes; fixes are rechecked.

The fastest accepted compatible runner that passes the model's gate becomes the deterministic default for that chip in the **next release**. Maintainers update `[selection]` and any accepted Exact/Fast kernel entries; users do not race candidates at Get or Load. A failed or incomplete attempt remains in the log, never silently replaces the default. Results from a different suite, profile, dtype or mode are not evidence that one runner beat another.

Before release, maintainers validate/commit folders, build/sign the bundled catalog, verify catalog inventories and candidate packaging, and report bundle size/idle behavior. Signing keys stay outside Git. Local development contributors rebase their task branch, rerun relevant checks and deliver with an ff-only merge under the coordinator's ownership rules. Do not mutate main directly or publish as cleanup.

To validate a cleanup without touching the signed seed, commit the source and run `Arsenal catalog build --output lab/catalog-candidate.json`, then `Arsenal catalog verify`. The output is unsigned; release signing remains a separate action.
