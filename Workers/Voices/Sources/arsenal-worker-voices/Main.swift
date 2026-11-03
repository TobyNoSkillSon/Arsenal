// SPDX-License-Identifier: MIT
import VoicesWorker

// Standalone voices worker for this package's own tests and wire checks; the app ships Workers/Native instead.
@main enum VoicesMain {
    static func main() async { await VoicesWorker.run() }
}
