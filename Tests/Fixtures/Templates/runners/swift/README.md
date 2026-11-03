<!-- SPDX-License-Identifier: MIT -->
# Swift CPU runner

From the repository root:

```sh
scripts/mlock work runner-swift-build -- env DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  swiftc -swift-version 6 Tests/Fixtures/Templates/runners/swift/main.swift -o /tmp/arsenal-toy-swift
scripts/mlock work runner-swift-check -- .build/debug/Arsenal runner-check \
  /tmp/arsenal-toy-swift --model Tests/Fixtures/Templates/runners/toy
```

Foundation only; no MLX or package dependencies. The reader blocks on stdin, one queue runs CPU arithmetic, and locks protect state and complete output frames. For production Swift workers, use `ArsenalWire` instead of duplicating framing.
