// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

@main enum StatsCommand {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        exit(args.first == "intake" ? await ModelIntakeCommand.run(arguments: Array(args.dropFirst())) : await PrecisionStatsCommand.run(arguments: args))
    }
}
