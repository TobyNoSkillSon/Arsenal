// SPDX-License-Identifier: MIT
import Darwin
import Foundation
import ChatWorker
import D1Worker
import ElephantWorker
import DecisionsWorker
import FluxWorker
import BiRefNetWorker
import VoicesWorker
import SoundWorker
import LanguageWorker
import EmbedWorker
import AudioWorker
import MusicWorker
import RestorationWorker
import PerceptionWorker
import GeometryWorker
import QwenImageWorker

// The family is the name the binary was started under: the app ships Helpers/arsenal-worker-native plus one symlink
// per family (arsenal-worker-voices, arsenal-worker-chat), and core launches arsenal-worker-<family>. One mechanism on
// purpose: voices re-executes its own path (symlink name kept) for its kernel self-test child, so a flag would be lost.
@main enum NativeWorker {
    static func main() async {
        let name = URL(fileURLWithPath: CommandLine.arguments.first ?? "").lastPathComponent
        if let kit = kitWorkers[name] { kit() }
        switch name {
        case "arsenal-worker-language": LanguageWorker.run()
        case "arsenal-worker-embed": EmbedWorker.run()
        case "arsenal-worker-audio": AudioWorker.run()
        case "arsenal-worker-music": MusicWorker.run()
        case "arsenal-worker-restoration": RestorationWorker.run()
        case "arsenal-worker-perception": PerceptionWorker.run()
        case "arsenal-worker-geometry": GeometryWorker.run()
        case "arsenal-worker-qwenimage": QwenImageWorker.run()
        case "arsenal-worker-voices": await VoicesWorker.run()
        case "arsenal-worker-chat": ChatWorker.run()
        case "arsenal-worker-sound", "arsenal-worker-enhancement": await SoundWorker.run()
        case "arsenal-worker-decisions": DecisionsWorker.run()
        case "arsenal-worker-d1": D1Worker.run()
        case "arsenal-worker-elephant": ElephantWorker.run()
        case "arsenal-worker-flux": await FluxWorker.run()
        case "arsenal-worker-birefnet": BiRefNetWorker.run()
        default:
            FileHandle.standardError.write(
                Data("\(name): start this worker through its family link (arsenal-worker-<family>, one per family in scripts/pkg-lib.sh)\n".utf8))
            exit(64)
        }
    }
}
