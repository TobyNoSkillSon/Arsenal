// SPDX-License-Identifier: MIT
import CoreServices
import Foundation

/// Watches `<support>/local-models/` so a model folder dropped in (or removed) appears in an open table without a click.
/// One FSEvents stream, no timer: it costs nothing while the folder is quiet.
///
/// A folder of weights takes a while to copy, and a rescan meanwhile would hash files still being written (and show
/// "local file changed during discovery"). So `changed` fires only once the tree has stopped changing: after an event,
/// the watcher compares the tree's stamp (paths, sizes, modification times) `settle` seconds apart, and every further
/// event restarts that wait. A copy of any length is one callback, after it ends. The callback only asks for the same
/// local rescan Refresh does; validation, consent and the read-only snapshot are unchanged.
public final class LocalFolderWatcher: @unchecked Sendable {
    public static let latency: CFTimeInterval = 1
    public let root: URL
    public let settle: TimeInterval
    private let changed: @Sendable () -> Void
    private let queue = DispatchQueue(label: "dev.arsenal.local-folders")
    /// The stream, guarded: start and stop may come from any thread.
    private let lock = NSLock()
    private var stream: FSEventStreamRef?
    /// Queue-confined: the pending settle check and the stamp it compares against.
    private var pending: DispatchWorkItem?
    private var lastStamp: Int?

    /// Creates `root` when missing, so the folder Open Arsenal Files shows is there to drop into.
    public init(root: URL, settle: TimeInterval = 2, changed: @escaping @Sendable () -> Void) throws {
        self.root = root
        self.settle = settle
        self.changed = changed
        queue.setSpecific(key: Self.queueKey, value: ObjectIdentifier(self))
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    deinit { stop() }

    public func start() {
        lock.withLock {
            guard stream == nil else { return }
            // FSEvents holds its own reference to the watcher for the stream's life: a callback never outlives it.
            var context = FSEventStreamContext(
                version: 0, info: Unmanaged.passUnretained(self).toOpaque(),
                retain: { info in info.map { UnsafeRawPointer(Unmanaged<LocalFolderWatcher>.fromOpaque($0).retain().toOpaque()) } },
                release: { info in info.map { Unmanaged<LocalFolderWatcher>.fromOpaque($0).release() } }, copyDescription: nil)
            let callback: FSEventStreamCallback = { _, info, _, _, _, _ in
                guard let info else { return }
                Unmanaged<LocalFolderWatcher>.fromOpaque(info).takeUnretainedValue().eventArrived()
            }
            guard
                let created = FSEventStreamCreate(
                    kCFAllocatorDefault, callback, &context, [root.path] as CFArray, FSEventStreamEventId(kFSEventStreamEventIdSinceNow), Self.latency,
                    FSEventStreamCreateFlags(kFSEventStreamCreateFlagWatchRoot))
            else { return }
            FSEventStreamSetDispatchQueue(created, queue)
            FSEventStreamStart(created)
            stream = created
        }
    }

    /// Stops the stream and drops a pending check; no callback runs after it returns.
    public func stop() {
        let stopped: FSEventStreamRef? = lock.withLock {
            defer { stream = nil }
            return stream
        }
        guard let stopped else { return }
        FSEventStreamStop(stopped)
        FSEventStreamInvalidate(stopped)
        FSEventStreamRelease(stopped)
        let drain = {
            self.pending?.cancel(); self.pending = nil
        }
        if DispatchQueue.getSpecific(key: Self.queueKey) == ObjectIdentifier(self) { drain() } else { queue.sync(execute: drain) }
    }
    public var running: Bool { lock.withLock { stream != nil } }
    private static let queueKey = DispatchSpecificKey<ObjectIdentifier>()

    /// On the queue: (re)start the wait for the tree to settle.
    private func eventArrived() {
        pending?.cancel()
        lastStamp = Self.stamp(root)
        schedule()
    }
    private func schedule() {
        let item = DispatchWorkItem { [weak self] in self?.check() }
        pending = item
        queue.asyncAfter(deadline: .now() + settle, execute: item)
    }
    private func check() {
        guard running else { return }
        let now = Self.stamp(root)
        if now != lastStamp {
            lastStamp = now
            schedule()
            return
        }
        pending = nil
        changed()
    }

    /// Every entry's relative path, size and modification time, hashed: equal stamps mean nothing moved meanwhile.
    static func stamp(_ root: URL) -> Int {
        var hasher = Hasher()
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let walk = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) else { return 0 }
        var entries: [(String, Int, Double)] = []
        for case let url as URL in walk {
            let values = try? url.resourceValues(forKeys: Set(keys))
            entries.append((url.path, values?.fileSize ?? -1, values?.contentModificationDate?.timeIntervalSinceReferenceDate ?? 0))
        }
        for e in entries.sorted(by: { $0.0 < $1.0 }) { hasher.combine(e.0); hasher.combine(e.1); hasher.combine(e.2) }
        return hasher.finalize()
    }
}
