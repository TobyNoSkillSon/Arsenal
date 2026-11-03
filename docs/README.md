# Arsenal documentation

- [MCP](MCP.md) and [HTTP API](API.md): connect agents and apps, discover operations, acquire models and run jobs.
- [Integration checklist](INTEGRATING.md), [dependency policy](DEPENDENCIES.md), [model catalog](MODELS.md) and [tool-window contracts](TOOL-WINDOWS-PLAN.md).
- [Model folders](MODEL-FOLDER.md) and [Adding a model](ADDING-A-MODEL.md): declarations, local models and qualification.
- [Runners](RUNNERS.md): native process protocol and conformance checks.
- [Contributing](CONTRIBUTING.md), [Contracts](CONTRACTS.md), [Kernels](KERNELS.md) and [Benchmarks](BENCHMARKS.md): implementation and verification.
- [Menu tests](MENU-MOTION-TESTS.md) and [Updates](UPDATES.md): app lifecycle checks.
- [Agent skill](../skill/SKILL.md) and [project instructions](../AGENTS.md).

`Sources/` holds the app, services and commands; `Workers/` holds native engines; `Packages/` holds shared Swift dependencies. `models/` contains current model recipes, table results and quality bindings. `Resources/` contains the signed catalog, shared kinds and schemas. The installed agent skill stays in `skill/`. Build and test tooling lives in `scripts/`; regression inputs live in `Tests/Fixtures/`.

Ignored `lab/` holds experiments, raw intake captures, historical notes and research. A fresh clone builds and validates without it. Tests materialize frozen evidence fixtures in their private build directory; catalog validation checks current tracked bindings and checks raw evidence only when it is present.
