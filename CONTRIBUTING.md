# Contributing to BOUNTY

Please begin with [the architecture guide](./docs/architecture.md) and [extension guide](./docs/extension_guide.md). Numerical logic belongs in the typed Rust core, browser computation belongs behind the worker protocol, and system metadata belongs in the shared catalog. Avoid adding a parallel code path for one interface control.

Frontend changes must remain within the strict TypeScript baseline. Do not introduce `any`, suppress a compiler error, or edit generated `frontend/pkg` declarations to make a call compile. Add or refine the domain contract in `frontend/src/types/domain.ts`, validate external JSON or worker data at runtime, and keep component props narrower than the complete application state. Run `npm run typecheck` from `frontend/` while developing; the production build and repository verification repeat it.

Create a focused branch and keep changes small enough to validate. Add tests for new behavior, especially configuration validation, numerical invariants, worker messages, schema changes, and major components. Run the complete baseline before requesting review:

```bash
./scripts/verify.sh
```

For a numerical change, describe the mathematical quantity, parameters, expected behavior, residual or convergence evidence, and known limitation. If a committed scientific metric changes, explain why; do not refresh the reference file without investigating the difference. Attach an experiment JSON when it helps reproduce the result.

Update documentation in the same change when a control, supported system, architecture boundary, experiment schema, or scientific interpretation changes. Write directly and use precise terms. A tutorial should state the question, exact configuration, expected observation, validation check, and limitation.

Commit generated build outputs only when a release process explicitly requires them. `frontend/pkg`, `frontend/dist`, evaluation outputs, caches, and release archives are generated and ignored. The supported runtime is the Rust/WASM frontend; do not add features to the legacy Python prototype.
