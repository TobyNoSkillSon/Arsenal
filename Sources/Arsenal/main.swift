// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import ArsenalCommands

// Entry point.
//   Arsenal            the menu-bar app (ARSENAL_STUB=1: the stub catalog, no downloads, no GPU)
//   Arsenal stats …   headless held precision queue (list, seed, run --go)
//   Arsenal catalog …   headless model-folder validation and catalog tools
//   Arsenal runner-check …   headless runner protocol checks (runner remains a child process)
//   Arsenal login on|off|status   Launch at Login of the installed app
//   Arsenal --smoke    headless smoke on the stub catalog: status item, menu, table, hosted API; one line (scripts/build.sh)
// Offscreen renders and real cycles are test support, not part of the app: Tests/ArsenalAppTests/QA (QAHarness.swift).

// Line-buffered: a run killed by its alarm still leaves every printed line in the log.
setvbuf(stdout, nil, _IOLBF, 0)
// Reuse the app's linked Core. Headless commands neither open UI nor claim the user's app.
// Keep all dispatch before NSApplication/SingleInstance, including when the menu-bar app is running.
switch CommandLine.arguments.dropFirst().first {
case "stats":
    Task { exit(await PrecisionStatsCommand.run(arguments: Array(CommandLine.arguments.dropFirst(2)))) }
    dispatchMain()
case "catalog":
    Task { exit(await CatalogCommand.run(arguments: Array(CommandLine.arguments.dropFirst(2)))) }
    dispatchMain()
case "login":
    exit(LoginCommand.run(Array(CommandLine.arguments.dropFirst(2))))
case "runner-check":
    RunnerCheckCommand.run(arguments: Array(CommandLine.arguments.dropFirst(2)))
default: break
}
let application = NSApplication.shared
let delegate: NSObject & NSApplicationDelegate
if CommandLine.arguments.contains("--smoke") {
    delegate = SmokeDelegate()
} else {
    guard SingleInstance.claim() else { exit(0) }
    delegate = LaunchDelegate()
}
application.delegate = delegate
withExtendedLifetime(delegate) { application.run() }
exit((delegate as? ExitCoding)?.exitCode ?? 0)
