// SPDX-License-Identifier: MIT
// Protein chains: PDB reading and writing and the backbone frame geometry ESM3 uses, ported from
// esm/utils/structure/{protein_chain,affine3d}.py (esm commit 8f0834c). Foundation only, in Double.
import Foundation

typealias Vec3 = (Double, Double, Double)

/// One chain's backbone: per residue N, CA, C (nil where an atom is missing) and its one-letter sequence.
struct Backbone {
    var sequence: [Character]
    var atoms: [[Vec3?]]  // [L][3]: N, CA, C
    var residueNumbers: [Int]
    var chain: String
    var count: Int { sequence.count }

    /// Refuses a chain (or the 1-indexed segment of it) in which no residue has all of N, CA and C: such a file would
    /// encode to tokens for an empty backbone, or condition a prompt on no coordinates, without saying so.
    func requireBackbone(_ segment: ClosedRange<Int>? = nil) throws {
        let rows = segment.map { atoms[($0.lowerBound - 1) ..< $0.upperBound] } ?? atoms[...]
        guard rows.contains(where: { $0.allSatisfy { $0 != nil } }) else {
            throw TokenError("The structure has no residue with all backbone atoms (N, CA and C)\(segment.map { " in residues \($0.lowerBound)-\($0.upperBound)" } ?? ""); send a PDB with backbone coordinates.")
        }
    }
}

enum PDB {
    /// residue_constants.restype_3to1 (the 20 standard residues); any other amino acid reads as X.
    static let threeToOne: [String: Character] = [
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    ]
    static let oneToThree: [Character: String] = Dictionary(uniqueKeysWithValues: threeToOne.map { ($1, $0) })
    /// ATOM residues that are not amino acids: nucleotides, water and the common ions. Everything else in an ATOM record
    /// is read as a residue, a nonstandard one as X, as biotite's filter_amino_acids keeps modified residues (MSE,
    /// SEP, NLE, ...) and the maker's 3-to-1 map turns them into X; dropping them would shift every later position.
    static let notAminoAcids: Set<String> = [
        "A", "C", "G", "U", "T", "I", "N", "DA", "DC", "DG", "DT", "DU", "DI", "DN", "HOH", "WAT", "DOD", "H2O",
        "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "SO4", "PO4",
    ]

    /// A structure file's text: UTF-8, else ISO Latin-1 (PDB records are ASCII; older tools leave a stray byte such as a
    /// degree sign in a REMARK). A gzip file or one that cannot be read is the caller's to fix.
    static func read(_ url: URL) throws -> String {
        guard let data = try? Data(contentsOf: url) else { throw TokenError("structure could not be read; send a plain-text PDB file.") }
        guard !data.starts(with: [0x1F, 0x8B]) else { throw TokenError("structure is gzip-compressed (.gz); unzip it and send the plain-text PDB file.") }
        // Latin-1 maps every byte to a character, so the fallback always decodes.
        return String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) ?? ""
    }

    /// ProteinChain.from_pdb: the first model, ATOM records of amino acids only (HETATM excluded, as the maker filters
    /// hetero atoms), one chain (the first, or `chain`), the first alternate location of each atom.
    static func parse(_ text: String, chain wanted: String? = nil, maxResidues: Int) throws -> Backbone {
        var chainID: String? = wanted
        var order: [String] = [], names: [String: String] = [:], numbers: [String: Int] = [:]
        var atoms: [String: [String: Vec3]] = [:]
        var anyAtom = false
        for raw in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            let line = String(raw)
            if line.hasPrefix("ENDMDL") { if anyAtom { break } else { continue } }
            guard line.hasPrefix("ATOM  "), line.count >= 54 else { continue }
            let chars = Array(line)
            func field(_ a: Int, _ b: Int) -> String { String(chars[a ..< Swift.min(b, chars.count)]).trimmingCharacters(in: .whitespaces) }
            let resName = field(17, 20), lineChain = String(chars[21])
            guard !resName.isEmpty, !notAminoAcids.contains(resName) else { continue }
            if chainID == nil { chainID = lineChain }
            guard lineChain == chainID else { continue }
            guard let resSeq = Int(field(22, 26)), let x = Double(field(30, 38)), let y = Double(field(38, 46)), let z = Double(field(46, 54)) else {
                throw TokenError("PDB line is malformed: \(line.prefix(30))")
            }
            anyAtom = true
            let key = "\(resSeq)|\(chars[26])"
            if atoms[key] == nil {
                guard order.count < maxResidues else { throw TokenError("The structure has more than \(maxResidues) residues.") }
                order.append(key); names[key] = resName; numbers[key] = resSeq; atoms[key] = [:]
            }
            var name = field(12, 16)
            if name == "SE" && resName == "MSE" { name = "SD" }
            if atoms[key]![name] == nil { atoms[key]![name] = (x, y, z) }  // first alternate location
        }
        guard let chainID, !order.isEmpty else { throw TokenError("The PDB file has no amino-acid ATOM records\(wanted.map { " in chain \($0)" } ?? "").") }
        return Backbone(
            sequence: order.map { threeToOne[names[$0]!] ?? "X" },
            atoms: order.map { key in ["N", "CA", "C"].map { atoms[key]![$0] } },
            residueNumbers: order.map { numbers[$0]! }, chain: chainID)
    }

    /// PDB text of a backbone: N, CA, C and the inferred O (ProteinChain.infer_oxygen) per residue, B-factor = pLDDT x 100.
    static func write(sequence: [Character], atoms: [[Vec3]], plddt: [Double]?, title: String) -> String {
        var lines = ["REMARK   1 \(title)"]
        let oxygens = Geometry.inferOxygen(atoms)
        var serial = 1
        for (i, residue) in atoms.enumerated() {
            let resName = oneToThree[sequence[i]] ?? "UNK"
            let b = (plddt?[i] ?? 1) * 100
            var named: [(String, Vec3, String)] = [(" N  ", residue[0], "N"), (" CA ", residue[1], "C"), (" C  ", residue[2], "C")]
            if let o = oxygens[i] { named.append((" O  ", o, "O")) }
            for (name, p, element) in named {
                lines.append(String(format: "ATOM  %5d %@ %@ A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2@",
                                    serial, name, resName.padding(toLength: 3, withPad: " ", startingAt: 0), i + 1, p.0, p.1, p.2, 1.0, b, element))
                serial += 1
            }
        }
        lines.append(String(format: "TER   %5d      %@ A%4d", serial, oneToThree[sequence.last ?? "X"] ?? "UNK", atoms.count))
        lines.append("END")
        return lines.joined(separator: "\n") + "\n"
    }
}

/// Rigid frames (affine3d.py): rotation as columns [x, e1, e2] of Gram-Schmidt, translation at the origin atom.
struct Frame {
    var rotation: [Double]  // 3 x 3, row-major
    var translation: Vec3
    static let identity = Frame(rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: (0, 0, 0))

    func apply(_ p: Vec3) -> Vec3 {
        let r = rotation
        return (r[0] * p.0 + r[1] * p.1 + r[2] * p.2 + translation.0,
                r[3] * p.0 + r[4] * p.1 + r[5] * p.2 + translation.1,
                r[6] * p.0 + r[7] * p.1 + r[8] * p.2 + translation.2)
    }
}

enum Geometry {
    static func sub(_ a: Vec3, _ b: Vec3) -> Vec3 { (a.0 - b.0, a.1 - b.1, a.2 - b.2) }
    static func dot(_ a: Vec3, _ b: Vec3) -> Double { a.0 * b.0 + a.1 * b.1 + a.2 * b.2 }

    /// _graham_schmidt(x_axis, xy_plane, eps) -> rotation with columns x, e1, x cross e1.
    static func gramSchmidt(_ xAxis: Vec3, _ xyPlane: Vec3, eps: Double = 1e-10) -> [Double] {
        let xn = (dot(xAxis, xAxis) + eps).squareRoot()
        let x = (xAxis.0 / xn, xAxis.1 / xn, xAxis.2 / xn)
        let d = dot(x, xyPlane)
        var e1 = (xyPlane.0 - x.0 * d, xyPlane.1 - x.1 * d, xyPlane.2 - x.2 * d)
        let en = (dot(e1, e1) + eps).squareRoot()
        e1 = (e1.0 / en, e1.1 / en, e1.2 / en)
        let e2 = (x.1 * e1.2 - x.2 * e1.1, x.2 * e1.0 - x.0 * e1.2, x.0 * e1.1 - x.1 * e1.0)
        return [x.0, e1.0, e2.0, x.1, e1.1, e2.1, x.2, e1.2, e2.2]
    }

    /// Affine3D.from_graham_schmidt(neg_x_axis, origin, xy_plane).
    static func frame(negX: Vec3, origin: Vec3, xyPlane: Vec3) -> Frame {
        Frame(rotation: gramSchmidt(sub(origin, negX), sub(xyPlane, origin)), translation: origin)
    }

    /// build_affine3d_from_coordinates over N, CA, C rows (nil = missing residue or atom): the backbone frame of each
    /// residue with all three atoms finite, else the frame of the average backbone (identity rotation when no residue
    /// has coordinates). Returns the frames and the mask of residues with coordinates.
    static func backboneFrames(_ atoms: [[Vec3?]]) -> (frames: [Frame], mask: [Bool]) {
        let mask = atoms.map { row in row.allSatisfy { p in p.map { [$0.0, $0.1, $0.2].allSatisfy { $0.isFinite && $0 < 1e6 } } ?? false } }
        var sums: [Vec3] = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        var n = 0.0
        for (row, ok) in zip(atoms, mask) where ok {
            for k in 0 ..< 3 { sums[k] = (sums[k].0 + row[k]!.0, sums[k].1 + row[k]!.1, sums[k].2 + row[k]!.2) }
            n += 1
        }
        let avg = sums.map { ($0.0 / (n + 1e-8), $0.1 / (n + 1e-8), $0.2 / (n + 1e-8)) }
        var blackHole = frame(negX: avg[2], origin: avg[1], xyPlane: avg[0])
        if n == 0 { blackHole.rotation = Frame.identity.rotation }
        let frames = zip(atoms, mask).map { row, ok in ok ? frame(negX: row[2]!, origin: row[1]!, xyPlane: row[0]!) : blackHole }
        return (frames, mask)
    }

    /// ProteinChain.infer_oxygen: O from the frame (CA, C, next N); none for the last residue.
    static func inferOxygen(_ atoms: [[Vec3]]) -> [Vec3?] {
        let local: Vec3 = (0.6240, -1.0613, 0.0103)
        return atoms.indices.map { i in
            guard i + 1 < atoms.count else { return nil }
            let f = frame(negX: atoms[i][1], origin: atoms[i][2], xyPlane: atoms[i + 1][0])
            let o = f.apply(local)
            return [o.0, o.1, o.2].allSatisfy(\.isFinite) ? o : nil
        }
    }

    /// The structure head's local backbone (structure_proj.py BB_COORDINATES): N, CA, C in the residue frame.
    static let localBackbone: [Vec3] = [(0.5256, 1.3612, 0.0), (0, 0, 0), (-1.5251, 0, 0)]

    /// kNN graph of the structure encoder (misc.knn_graph): for each residue the `k` nearest by CA distance, residues
    /// without coordinates by sequence distance (100 per position past 1e6); ties keep the lower index (stable sort).
    static func knn(_ ca: [Vec3?], mask: [Bool], k: Int) -> [[Int]] {
        let L = ca.count, count = Swift.min(k, L)
        return (0 ..< L).map { i in
            let keys: [(Double, Int)] = (0 ..< L).map { j in
                if mask[i] && mask[j] {
                    let d = sub(ca[i]!, ca[j]!)
                    return (Double(Float(dot(d, d).squareRoot())), j)
                }
                return (Double(abs(i - j)) * 100 + 1e6, j)
            }
            return keys.sorted { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }.prefix(count).map(\.1)
        }
    }
}
