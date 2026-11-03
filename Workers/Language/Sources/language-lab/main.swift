import Foundation
import LanguageWorker
if CommandLine.arguments.count == 5, CommandLine.arguments[1] == "--inspect-tokenizer" {
    do {
        try LanguageWorker.inspectTokenizer(
            directory: URL(fileURLWithPath: CommandLine.arguments[2]), specification: URL(fileURLWithPath: CommandLine.arguments[3]),
            output: URL(fileURLWithPath: CommandLine.arguments[4]))
    } catch { FileHandle.standardError.write(Data((String(reflecting: error) + "\n").utf8)); exit(1) }
} else if CommandLine.arguments.count == 5, CommandLine.arguments[1] == "--inspect-header" {
    do {
        try LanguageWorker.inspectHeader(
            config: URL(fileURLWithPath: CommandLine.arguments[2]), header: URL(fileURLWithPath: CommandLine.arguments[3]), output: URL(fileURLWithPath: CommandLine.arguments[4]))
    } catch { FileHandle.standardError.write(Data((String(reflecting: error) + "\n").utf8)); exit(1) }
} else {
    LanguageWorker.run()
}
