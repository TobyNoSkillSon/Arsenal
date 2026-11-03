// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import ArsenalWire

/// posix_spawn establishes the process group before executing the worker; no setpgid race.
final class ChildProcess: @unchecked Sendable {
    let pid: pid_t
    private let asyncOutput: AsyncPipeReader
    private let asyncError: AsyncPipeReader
    private var stderrTask: Task<Void, Never>?
    private let groupLock = NSLock()
    private var reaped = false
    private let input: Int32
    private let writeLock = NSLock()
    private var inputClosed = false
    init(executable: URL, arguments: [String], environment: [String: String], log: URL) throws {
        var inPipe: [Int32] = [0, 0], outPipe: [Int32] = [0, 0], errPipe: [Int32] = [0, 0]
        guard pipe(&inPipe) == 0 else { throw WorkerFailure(code: "spawn", message: "Cannot create worker pipe.") }
        guard pipe(&outPipe) == 0 else { close(inPipe[0]); close(inPipe[1]); throw WorkerFailure(code: "spawn", message: "Cannot create worker pipe.") }
        guard pipe(&errPipe) == 0 else { for fd in inPipe + outPipe { close(fd) }; throw WorkerFailure(code: "spawn", message: "Cannot create worker log pipe.") }
        var ownedFDs = inPipe + outPipe + errPipe
        defer { for fd in ownedFDs { close(fd) } }
        try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        let stderrFD = open(log.path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0o600)
        guard stderrFD >= 0 else { throw WorkerFailure(code: "spawn", message: "Cannot open worker log.") }; ownedFDs.append(stderrFD)
        for fd in inPipe + outPipe + errPipe { _ = fcntl(fd, F_SETFD, FD_CLOEXEC) }
        var actions: posix_spawn_file_actions_t?; posix_spawn_file_actions_init(&actions); defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_adddup2(&actions, inPipe[0], STDIN_FILENO)
        posix_spawn_file_actions_adddup2(&actions, outPipe[1], STDOUT_FILENO)
        posix_spawn_file_actions_adddup2(&actions, errPipe[1], STDERR_FILENO)
        var attr: posix_spawnattr_t?; posix_spawnattr_init(&attr); defer { posix_spawnattr_destroy(&attr) }
        // The app ignores SIGTERM for its DispatchSource; that disposition survives exec unless reset.
        var defaults = sigset_t(), mask = sigset_t()
        sigfillset(&defaults); sigemptyset(&mask)
        posix_spawnattr_setsigdefault(&attr, &defaults)
        posix_spawnattr_setsigmask(&attr, &mask)
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK))
        posix_spawnattr_setpgroup(&attr, 0)
        let argv = ([executable.path] + arguments).map { strdup($0) } + [nil]
        let envp = environment.sorted { $0.key < $1.key }.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { for p in argv + envp { free(p) } }
        var child: pid_t = 0
        let status = argv.withUnsafeBufferPointer { a in envp.withUnsafeBufferPointer { e in posix_spawn(&child, executable.path, &actions, &attr, a.baseAddress!, e.baseAddress!) }
        }
        guard status == 0 else { throw WorkerFailure(code: "spawn", message: "Cannot start worker: \(String(cString: strerror(status))).") }
        pid = child; input = inPipe[1]
        ownedFDs.removeAll { $0 == inPipe[1] || $0 == outPipe[0] || $0 == errPipe[0] || $0 == stderrFD }
        let logInput = errPipe[0]
        asyncOutput = AsyncPipeReader(descriptor: outPipe[0])
        asyncError = AsyncPipeReader(descriptor: logInput)
        let asyncError = self.asyncError
        stderrTask = Task.detached {
            defer { asyncError.finish(); close(stderrFD) }
            var written = 0
            do {
                while let bytes = try await asyncError.read(count: 16_384) {
                    // Drain even after a failed disk write; inference must not depend on log storage.
                    if written + bytes.count > Self.logLimit {
                        guard ftruncate(stderrFD, 0) == 0, lseek(stderrFD, 0, SEEK_SET) >= 0 else { continue }
                        written = 0
                    }
                    bytes.withUnsafeBytes { raw in
                        var offset = 0
                        while offset < raw.count {
                            let n = Darwin.write(stderrFD, raw.baseAddress! + offset, raw.count - offset)
                            if n < 0 && errno == EINTR { continue }; if n <= 0 { break }; offset += n
                        }
                        written += offset
                    }
                }
            } catch {}
        }
        _ = fcntl(input, F_SETFL, fcntl(input, F_GETFL) | O_NONBLOCK)
        _ = fcntl(input, F_SETNOSIGPIPE, 1)
    }
    func write(_ frame: WireFrame, deadline: Date) throws {
        let data = try frame.encoded(); writeLock.lock(); defer { writeLock.unlock() }
        guard !inputClosed else { throw WorkerFailure(code: "worker_exit", message: "Worker has exited.") }
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                let left = deadline.timeIntervalSinceNow
                guard left > 0 else { throw WorkerFailure(code: "write_timeout", message: "Worker is not reading requests.") }
                var p = pollfd(fd: input, events: Int16(POLLOUT), revents: 0)
                let polled = poll(&p, 1, Int32(min(100, left * 1000)))
                if polled < 0 && errno == EINTR { continue }
                guard polled >= 0 else { throw WorkerFailure(code: "write", message: "Worker pipe failed.") }
                if polled == 0 { continue }
                let n = Darwin.write(input, bytes.baseAddress! + offset, bytes.count - offset)
                if n < 0 && [EINTR, EAGAIN].contains(errno) { continue }
                guard n > 0 else { throw WorkerFailure(code: "write", message: "Worker stopped reading requests.") }; offset += n
            }
        }
    }
    func closeInput() { writeLock.lock(); defer { writeLock.unlock() }; if !inputClosed { close(input); inputClosed = true } }
    static let logLimit = 2 * 1_048_576
    func signal(_ value: Int32) { groupLock.lock(); defer { groupLock.unlock() }; if !reaped { _ = kill(-pid, value) } }
    func awaitExit() -> Int32 {
        // Keep the leader waitable until group cleanup: its PID cannot be reused as a new process group.
        var info = siginfo_t()
        var waited: Int32
        repeat { waited = waitid(P_PID, id_t(pid), &info, WEXITED | WNOWAIT) } while waited == -1 && errno == EINTR
        groupLock.lock()
        if waited == 0 { _ = kill(-pid, SIGKILL) }
        var status: Int32 = 0
        while waitpid(pid, &status, 0) == -1 && errno == EINTR {}
        reaped = true
        groupLock.unlock()
        // Let final frames/log bytes drain. An escaped descendant retaining a pipe cannot hold readers forever.
        let output = asyncOutput, error = asyncError
        Task.detached {
            try? await Task.sleep(for: .milliseconds(200)); output.finish(); error.finish()
        }
        return status
    }
    /// A process event wakes only after exit; waitid/waitpid never parks a pool thread.
    func awaitExitAsync() async -> Int32 {
        await withCheckedContinuation { reply in
            let source = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: DispatchQueue.global(qos: .utility))
            source.setEventHandler {
                let status = self.awaitExit()
                // Cancellation runs after the event handler returns. Release its capture of the child before
                // resuming the caller, so a completed await also means the exit source no longer owns it.
                source.setEventHandler(handler: nil)
                source.setCancelHandler { reply.resume(returning: status) }
                source.cancel()
            }
            source.resume()
        }
    }
    func readOutputAsync(count: Int) async throws -> Data? { try await asyncOutput.read(count: count) }
    func finishOutput() { asyncOutput.finish() }
    func readPacket() async throws -> (frame: WireFrame, rawJSON: Data?)? {
        func exact(_ count: Int, eof: Bool = false) async throws -> Data? {
            var result = Data()
            while result.count < count {
                guard let bytes = try await readOutputAsync(count: count - result.count) else {
                    if result.isEmpty && eof { return nil }; throw WireError.truncatedFrame
                }
                result.append(bytes)
            }
            return result
        }
        do {
            guard let header = try await exact(4, eof: true) else { return nil }
            let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            guard length <= WireFrame.maximumLength else { throw WireError.oversizedFrame }
            guard length > 0, let body = try await exact(Int(length)) else { throw WireError.malformedFrame }
            return (try WireFrame.decodeBody(body), body.first == 1 ? Data(body.dropFirst()) : nil)
        } catch { finishOutput(); throw error }
    }
    func finishReaders() async { await stderrTask?.value }
    deinit { closeInput() }
}
