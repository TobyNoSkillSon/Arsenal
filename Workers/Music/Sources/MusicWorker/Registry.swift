// SPDX-License-Identifier: MIT
import IntegrationKit

/// Owned by the music integrator. Register adapters here; no changes to the HTTP API, pins or native dispatcher.
public enum MusicWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["yue2"], conversion: MusicConversion.yueBase) { YuEAdapter() },
            NativeRegistration([], conversion: MusicConversion.yueTiers) { YuEAdapter() },
            NativeRegistration(["ace_step", "acestep"], conversion: MusicConversion.aceBase) { ACEAdapter() },
            NativeRegistration([], conversion: MusicConversion.aceTiers) { ACEAdapter() },
            NativeRegistration(["minimax_music3"], conversion: MusicConversion.miniMaxBase) { MiniMaxAdapter() }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(family: "music", kinds: ["sound"], operations: ["generate_music"], registrations: registrations)
    }
}
