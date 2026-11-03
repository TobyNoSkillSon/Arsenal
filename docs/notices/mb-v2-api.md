<!-- SPDX-License-Identifier: MIT -->
# mb-v2-api test dependencies

`openapi-spec-validator` 0.7.2 (BSD-3-Clause) validates generated OpenAPI 3.1 in the private SDK test environment only. Its hash-pinned dependency graph is in `tests/conformance/requirements.lock`, including `openapi-schema-validator` (BSD-3-Clause), `jsonschema` (MIT), `jsonschema-specifications` (MIT), `referencing` (MIT), `rpds-py` (MIT), `jsonschema-path` (Apache-2.0), `lazy-object-proxy` (BSD-2-Clause) and `pathable` (Apache-2.0). None is bundled into the app.
