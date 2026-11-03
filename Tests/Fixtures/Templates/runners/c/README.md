<!-- SPDX-License-Identifier: MIT -->
# C CPU runner

From the repository root:

```sh
scripts/mlock work runner-c-build -- clang -std=c11 -O2 -Wall -Wextra -Werror -pthread \
  Tests/Fixtures/Templates/runners/c/main.c -o /tmp/arsenal-toy-c
scripts/mlock work runner-c-check -- .build/debug/Arsenal runner-check \
  /tmp/arsenal-toy-c --model Tests/Fixtures/Templates/runners/toy
```

Plain CPU arithmetic with POSIX threads and macOS task memory reporting. No MLX, Python, Metal compiler or third-party library. This is the non-MLX engine example: replacing arithmetic with custom Metal/Core ML/C++/Rust work changes the engine, not the wire contract. The small JSON tokenizer is for this fixture, not a recommended general-purpose production JSON dependency.
