// SPDX-License-Identifier: MIT
// System One over the MLX forward, with the vendor runner's request semantics (`runner.py`): a request with one
// question and no pictures is one chain; any other request is one tree, its state (and pictures) the trunk and its
// questions the branches. Chains and trees are independent, so several share one pass up to a token budget.
import Foundation
import MLX

public struct Cancelled: Error {}

/// One request: a state (nil = none, pictures alone) and its questions, with decoded pictures.
public struct Request {
    public let state: PyJSON?
    public let questions: [Question]
    public let pictures: [Picture]
    public init(state: PyJSON?, questions: [Question], pictures: [Picture] = []) {
        self.state = state
        self.questions = questions
        self.pictures = pictures
    }
}

/// Each question's option probabilities and the tokens the request read, or the error that request alone raised.
public typealias Outcome = Result<(probabilities: [[Double]], read: Int), Error>

public final class SystemOne: @unchecked Sendable {
    public let tokenizer: LFMTokenizer
    public let verbalizer: Verbalizer
    public let model: D1Model
    let store: SafetensorsFile
    let weights: URL
    let bos: String
    public let tokenBudget: Int
    public let processor: PictureProcessor
    private var tower: VisionTower?

    public init(weights: URL, tokenBudget: Int = 4096) throws {
        self.weights = weights
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: weights.appendingPathComponent("tokenizer_config.json"))) as? [String: Any]
        bos = config?["bos_token"] as? String ?? ""
        // The tokenizer's 18 MB JSON parses on another thread while the language model loads.
        let tokenizerURL = weights.appendingPathComponent("tokenizer.json")
        final class Box: @unchecked Sendable { var value: Result<LFMTokenizer, Error>? }
        let parsed = Box()
        let done = DispatchSemaphore(value: 0)
        Thread {
            parsed.value = Result { try LFMTokenizer(url: tokenizerURL) }
            done.signal()
        }.start()
        processor = try PictureProcessor(config: weights.appendingPathComponent("processor_config.json"))
        let store = try SafetensorsFile(url: weights.appendingPathComponent("model.safetensors"))
        self.store = store
        let loaded = Result { try D1Model(weights: weights, store: store) }
        done.wait()
        tokenizer = try parsed.value!.get()
        model = try loaded.get()
        verbalizer = Verbalizer(tokenizer: tokenizer)
        self.tokenBudget = tokenBudget
    }

    public func loadPictures(_ urls: [URL], budget: inout PictureBudget) throws -> [Picture] {
        try processor.loadCapped(urls, budget: &budget)
    }

    // MARK: prompt

    public func render(_ state: PyJSON?, _ q: Question) throws -> String { Prompt.prefix(state, bos: bos) + (try verbalizer.suffix(q)) }

    /// The longest prompt one of `questions` makes over a text state (vendor `SystemOne.tokens`).
    public func tokens(_ state: PyJSON?, _ questions: [Question]) throws -> Int {
        let prefix = tokenizer.encode(Prompt.prefix(state, bos: bos)).count
        return prefix + (try questions.map { tokenizer.encode(try verbalizer.suffix($0)).count }.max() ?? 0)
    }

    /// Processor-expanded prompt length with `pictures` and the longest question (the vendor adapter's check).
    public func imageTokens(_ state: PyJSON?, _ questions: [Question], _ pictures: [Picture]) throws -> Int {
        let prefix = Prompt.prefix(state, bos: bos, images: PictureProcessor.markup(pictures.count))
        let suffixes = try questions.map { try verbalizer.suffix($0) }
        let longest = suffixes.max { tokenizer.encode($0).count < tokenizer.encode($1).count } ?? ""
        let layouts = pictures.map { processor.layout(processor.cap($0)) }
        return tokenizer.encode(try processor.expand(prefix + longest, layouts)).count
    }

    /// One chain and one small tree, so the kernels are built at load rather than in the first request.
    public func warm() throws {
        let q = Question(kind: .noul, instructions: .string("Is this a test?"))
        let c = Question(kind: .choice, instructions: .string("Which?"), options: [("a", .string("first")), ("b", .string("second"))])
        _ = try run([Request(state: .string("warm-up"), questions: [q]), Request(state: .string("warm-up"), questions: [q, c])])
    }

    // MARK: passes

    struct Entry {
        let owner: Int
        let first: Int
        let trunk: [Int]
        let branches: [[Int]]
        let questions: [Question]
        var tokens: Int { trunk.count + branches.reduce(0) { $0 + $1.count } }
    }

    struct Launched {
        let entries: [Entry]
        let wanted: [Int]
        let logits: MLXArray
    }

    /// Queues one MLX pass over `entries`; `read` turns it into probabilities.
    private func launch(_ entries: [Entry], embeds: ((Layout) -> MLXArray)? = nil) throws -> Launched {
        var layout = Layout()
        for e in entries { layout.add(trunk: e.trunk, branches: e.branches) }
        var wanted = Set<Int>()
        for e in entries { for q in e.questions { for g in try verbalizer.readoutIDs(q) { wanted.formUnion(g) } } }
        let sorted = wanted.sorted()
        let logits = model.logits(model.hidden(layout, embeds: embeds?(layout)), sorted)
        asyncEval(logits)
        return Launched(entries: entries, wanted: sorted, logits: logits)
    }

    private func read(_ l: Launched) throws -> [[[Double]]] {
        let values = l.logits.asArray(Float.self)
        let columns = l.wanted.count
        var column: [Int: Int] = [:]
        for (j, t) in l.wanted.enumerated() { column[t] = j }
        var out: [[[Double]]] = []
        var row = 0
        for e in l.entries {
            var probs: [[Double]] = []
            for q in e.questions {
                let base = row * columns
                probs.append(Readout.probabilities(try verbalizer.readoutIDs(q)) { Double(values[base + column[$0]!]) })
                row += 1
            }
            out.append(probs)
        }
        return out
    }

    /// Consecutive groups of at most `tokenBudget` tokens (a lone larger entry is its own group).
    private func plan(_ sizes: [Int]) -> [[Int]] {
        var out: [[Int]] = []
        var used = 0
        for (i, n) in sizes.enumerated() {
            if !out.isEmpty && used + n <= tokenBudget {
                out[out.count - 1].append(i)
                used += n
            } else {
                out.append([i])
                used = n
            }
        }
        return out
    }

    /// A request's tree as entries whose branches total at most `tokenBudget` tokens; each repeats the trunk. A lone
    /// branch is a plain chain (trunk + branch), the same arithmetic without a branch mask.
    private func split(owner: Int, trunk: [Int], branches: [[Int]], questions: [Question]) -> [Entry] {
        var out: [Entry] = []
        var group: [Int] = []
        var used = 0
        func emit() {
            if group.count == 1 {
                out.append(Entry(owner: owner, first: group[0], trunk: trunk + branches[group[0]], branches: [], questions: [questions[group[0]]]))
            } else {
                out.append(Entry(owner: owner, first: group[0], trunk: trunk, branches: group.map { branches[$0] }, questions: group.map { questions[$0] }))
            }
        }
        for (j, b) in branches.enumerated() {
            if !group.isEmpty && used + b.count > tokenBudget {
                emit()
                group = []
                used = 0
            }
            group.append(j)
            used += b.count
        }
        emit()
        return out
    }

    /// Each request's outcome. Text passes are queued one ahead of the one being read, so the host prepares the next
    /// pass while the GPU runs the current one. A pass that fails is retried entry by entry, so one bad item cannot take
    /// the others down. `cancelled` is checked before every pass.
    public func run(_ requests: [Request], cancelled: () -> Bool = { false }) throws -> [Outcome] {
        var entries: [Entry] = []
        var readCount: [Int: Int] = [:]
        var errors: [Int: Error] = [:]
        var probs: [Int: [[Double]?]] = [:]
        for (i, r) in requests.enumerated() where r.pictures.isEmpty {
            probs[i] = Array(repeating: nil, count: r.questions.count)
            do {
                if r.questions.count == 1 {
                    let row = tokenizer.encode(try render(r.state, r.questions[0]))
                    entries.append(Entry(owner: i, first: 0, trunk: row, branches: [], questions: r.questions))
                    readCount[i] = row.count
                    continue
                }
                let trunk = tokenizer.encode(Prompt.prefix(r.state, bos: bos))
                let branches = try r.questions.map { tokenizer.encode(try verbalizer.suffix($0)) }
                entries += split(owner: i, trunk: trunk, branches: branches, questions: r.questions)
                readCount[i] = trunk.count + branches.reduce(0) { $0 + $1.count }
            } catch {
                errors[i] = error
            }
        }
        func store(_ e: Entry, _ value: Result<[[Double]], Error>) {
            switch value {
            case .success(let p): for (k, v) in p.enumerated() { probs[e.owner]![e.first + k] = v }
            case .failure(let error): if errors[e.owner] == nil { errors[e.owner] = error }
            }
        }
        func finish(_ group: [Int], _ launched: Launched?) throws {
            if let launched, let values = try? read(launched) {
                for (j, v) in zip(group, values) { store(entries[j], .success(v)) }
                return
            }
            for j in group { // alone, so only the entry that cannot run fails
                if cancelled() { throw Cancelled() }
                do { store(entries[j], .success(try read(try launch([entries[j]]))[0])) } catch { store(entries[j], .failure(error)) }
            }
        }
        var pending: ([Int], Launched?)?
        for group in plan(entries.map(\.tokens)) {
            if cancelled() { throw Cancelled() }
            let launched = (group, try? launch(group.map { entries[$0] }))
            if let p = pending { try finish(p.0, p.1) }
            pending = launched
        }
        if let p = pending { try finish(p.0, p.1) }
        var out: [Outcome] = []
        for (i, r) in requests.enumerated() {
            if let error = errors[i] {
                out.append(.failure(error))
            } else if !r.pictures.isEmpty {
                if cancelled() { throw Cancelled() }
                do { out.append(.success(try pictureRequest(r, cancelled: cancelled))) } catch let c as Cancelled { throw c } catch {
                    out.append(.failure(error))
                }
            } else {
                out.append(.success((probs[i]!.map { $0! }, readCount[i]!)))
            }
        }
        return out
    }

    // MARK: pictures

    public func visionTower() throws -> VisionTower {
        if let tower { return tower }
        let t = try VisionTower(store: store, processor: processor)
        tower = t
        return t
    }

    private func pictureRequest(_ r: Request, cancelled: () -> Bool) throws -> (probabilities: [[Double]], read: Int) {
        let tower = try visionTower()
        let pics = r.pictures.map { processor.cap($0) }
        let layouts = pics.map { processor.layout($0) }
        let prefix = Prompt.prefix(r.state, bos: bos, images: PictureProcessor.markup(pics.count))
        let suffixes = try r.questions.map { try verbalizer.suffix($0) }
        let ids: [Int]
        let entries: [Entry]
        let tokensRead: Int
        if r.questions.count == 1 { // the whole prompt in one plain pass, as the vendor
            ids = tokenizer.encode(try processor.expand(prefix + suffixes[0], layouts))
            entries = [Entry(owner: 0, first: 0, trunk: ids, branches: [], questions: r.questions)]
            tokensRead = ids.count
        } else {
            ids = tokenizer.encode(try processor.expand(prefix, layouts))
            let branches = suffixes.map { tokenizer.encode($0) }
            entries = split(owner: 0, trunk: ids, branches: branches, questions: r.questions)
            tokensRead = ids.count + branches.reduce(0) { $0 + $1.count }
        }
        let slots = ids.indices.filter { ids[$0] == model.imageTokenID }
        let features = tower.features(Array(zip(pics, layouts)))
        guard slots.count == features.dim(0) else { throw JSONError("image placeholder count does not match the image features") }
        let slotArray = MLXArray(slots.map { Int32($0) })
        let embeds = { (layout: Layout) -> MLXArray in // one tree per pass, so the picture slots sit where they sit in `ids`
            let h = self.model.embed[MLXArray(layout.ids)]
            h[slotArray] = features.asType(h.dtype)
            return h
        }
        var probs = [[Double]](repeating: [], count: r.questions.count)
        for e in entries {
            if cancelled() { throw Cancelled() }
            let p = try read(try launch([e], embeds: embeds))[0]
            for (k, v) in p.enumerated() { probs[e.first + k] = v }
        }
        return (probs, tokensRead)
    }
}
