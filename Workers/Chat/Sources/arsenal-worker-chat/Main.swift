// SPDX-License-Identifier: MIT
import ChatWorker

// Standalone chat worker for this package's own tests and wire checks; the app ships Workers/Native instead.
@main struct ChatMain {
    static func main() { ChatWorker.run() }
}
