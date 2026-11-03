# Official Swift MCP SDK

Source: https://github.com/modelcontextprotocol/swift-sdk/tree/a0ae212ebf6eab5f754c3129608bc5557637e605
Version: 0.12.1, commit a0ae212ebf6eab5f754c3129608bc5557637e605. Library source only; upstream conformance executables, NIO and doc plugins are omitted. Full upstream licence is in LICENSE.

Arsenal changes: StatefulHTTPServerTransport retains at most 256 replay events / 4 MiB, drops oldest replay events, and clears terminated standalone GET continuations with a generation guard. Protocol models and client behaviour are unchanged.
