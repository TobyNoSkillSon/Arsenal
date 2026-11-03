// SPDX-License-Identifier: MIT
import Testing

// Only protocol timing probes serialize; the rest of the full Core suite remains parallel.
@Suite(.serialized) struct RuntimeTimingTests {}
