# Intake regression fixtures

Each model JSON maps original relative paths to exact UTF-8 file contents. The suites materialize these immutable inputs over a private copy of the current model folder, exercise the ordinary validators, and remove the tree on exit. They never read `lab/`.

These are selected structural and numerical receipts used by the existing regression tests: frozen bounds, qualification plans, source and snapshot identities, historical result cases and coverage receipts. Raw tensor/activation captures and unreferenced runs remain local in `lab/evidence/`. Current quality bindings and table statistics stay in `models/`.
