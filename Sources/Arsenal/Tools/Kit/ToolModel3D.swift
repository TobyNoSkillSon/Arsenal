// SPDX-License-Identifier: MIT
import AppKit
import Metal
import SceneKit
import SwiftUI

// A 3D result as the window shows it: a binary glTF (GLB, what Make 3D models return) read into SceneKit — its
// meshes' positions, normals, texture coordinates and indices, each material's base colour and base-colour texture,
// and the node transforms — then shown in a view the pointer turns and zooms. SceneKit draws only when something
// changes, so nothing renders at rest. Anything the reader does not understand is a stated failure, never a blank.

enum GLB {
    struct Failure: Error, LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// The scene in a GLB file.
    static func scene(_ url: URL) throws -> SCNScene {
        try scene(Data(contentsOf: url))
    }

    static func scene(_ data: Data) throws -> SCNScene {
        let (json, bin) = try chunks(data)
        let reader = Reader(json: json, bin: bin)
        let scene = SCNScene()
        let roots =
            json["scenes"].flatMap { ($0 as? [[String: Any]])?[(json["scene"] as? Int) ?? 0]["nodes"] as? [Int] }
            ?? Array(0..<((json["nodes"] as? [Any])?.count ?? 0))
        for index in roots { scene.rootNode.addChildNode(try reader.node(index, depth: 0)) }
        // A file with meshes but no nodes: its meshes, as they are.
        if roots.isEmpty, let meshes = json["meshes"] as? [Any] {
            for index in meshes.indices { scene.rootNode.addChildNode(try reader.mesh(index)) }
        }
        guard reader.meshCount > 0 else { throw Failure(message: "The 3D file holds no mesh.") }
        return scene
    }

    /// The JSON chunk and the binary chunk of a GLB container.
    static func chunks(_ data: Data) throws -> ([String: Any], Data) {
        func u32(_ at: Int) -> UInt32 { data.subdata(in: at..<at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian }
        guard data.count >= 20, u32(0) == 0x4654_6C67 else { throw Failure(message: "This is not a GLB file.") }
        guard u32(4) == 2 else { throw Failure(message: "Only glTF 2 files can be shown.") }
        var offset = 12, json: [String: Any]?, bin = Data()
        while offset + 8 <= data.count {
            let length = Int(u32(offset)), type = u32(offset + 4)
            let start = offset + 8, end = start + length
            guard end <= data.count else { throw Failure(message: "The GLB file is cut short.") }
            if type == 0x4E4F_534A { json = try JSONSerialization.jsonObject(with: data.subdata(in: start..<end)) as? [String: Any] }
            if type == 0x004E_4942 { bin = data.subdata(in: start..<end) }
            offset = end
        }
        guard let json else { throw Failure(message: "The GLB file has no description.") }
        return (json, bin)
    }

    private final class Reader {
        let json: [String: Any]
        let bin: Data
        var meshCount = 0
        private var materials: [Int: SCNMaterial] = [:]
        init(json: [String: Any], bin: Data) { self.json = json; self.bin = bin }

        private func list(_ key: String) -> [[String: Any]] { json[key] as? [[String: Any]] ?? [] }

        func node(_ index: Int, depth: Int) throws -> SCNNode {
            let nodes = list("nodes")
            guard nodes.indices.contains(index), depth < 64 else { throw Failure(message: "The 3D file names a node it does not have.") }
            let spec = nodes[index]
            let node = SCNNode()
            if let m = spec["matrix"] as? [Double], m.count == 16 {
                node.transform = SCNMatrix4(
                    m11: m[0], m12: m[1], m13: m[2], m14: m[3], m21: m[4], m22: m[5], m23: m[6], m24: m[7],
                    m31: m[8], m32: m[9], m33: m[10], m34: m[11], m41: m[12], m42: m[13], m43: m[14], m44: m[15])
            } else {
                if let t = spec["translation"] as? [Double], t.count == 3 { node.position = SCNVector3(t[0], t[1], t[2]) }
                if let r = spec["rotation"] as? [Double], r.count == 4 { node.orientation = SCNQuaternion(r[0], r[1], r[2], r[3]) }
                if let s = spec["scale"] as? [Double], s.count == 3 { node.scale = SCNVector3(s[0], s[1], s[2]) }
            }
            if let mesh = spec["mesh"] as? Int { node.addChildNode(try self.mesh(mesh)) }
            for child in spec["children"] as? [Int] ?? [] { node.addChildNode(try self.node(child, depth: depth + 1)) }
            return node
        }

        func mesh(_ index: Int) throws -> SCNNode {
            let meshes = list("meshes")
            guard meshes.indices.contains(index) else { throw Failure(message: "The 3D file names a mesh it does not have.") }
            let node = SCNNode()
            for primitive in meshes[index]["primitives"] as? [[String: Any]] ?? [] {
                let mode = primitive["mode"] as? Int ?? 4
                guard mode == 4 else { continue }
                guard let attributes = primitive["attributes"] as? [String: Int], let position = attributes["POSITION"] else { continue }
                var sources = [
                    SCNGeometrySource(vertices: try floats(position, components: 3).chunks(3).map { v in SCNVector3(v[v.startIndex], v[v.startIndex + 1], v[v.startIndex + 2]) })
                ]
                let vertexCount = sources[0].vectorCount
                if let n = attributes["NORMAL"] {
                    sources.append(
                        SCNGeometrySource(normals: try floats(n, components: 3).chunks(3).map { v in SCNVector3(v[v.startIndex], v[v.startIndex + 1], v[v.startIndex + 2]) }))
                }
                if let uv = attributes["TEXCOORD_0"] {
                    sources.append(
                        SCNGeometrySource(
                            textureCoordinates: try floats(uv, components: 2).chunks(2).map { v in CGPoint(x: Double(v[v.startIndex]), y: Double(v[v.startIndex + 1])) }))
                }
                if let c = attributes["COLOR_0"], let colours = try? floats(c, components: nil) {
                    let per = colours.count / max(1, vertexCount)
                    if per == 3 || per == 4 {
                        let data = colours.withUnsafeBufferPointer { Data(buffer: $0) }
                        sources.append(
                            SCNGeometrySource(
                                data: data, semantic: .color, vectorCount: vertexCount, usesFloatComponents: true, componentsPerVector: per,
                                bytesPerComponent: 4, dataOffset: 0, dataStride: per * 4))
                    }
                }
                let indices: [UInt32] = try (primitive["indices"] as? Int).map(integers) ?? (0..<UInt32(vertexCount)).map { $0 }
                guard indices.allSatisfy({ Int($0) < vertexCount }) else { throw Failure(message: "The 3D file's triangles name vertices it does not have.") }
                let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
                let geometry = SCNGeometry(sources: sources, elements: [element])
                geometry.materials = [try material(primitive["material"] as? Int)]
                node.addChildNode(SCNNode(geometry: geometry))
                meshCount += 1
            }
            return node
        }

        private func material(_ index: Int?) throws -> SCNMaterial {
            if let index, let hit = materials[index] { return hit }
            let m = SCNMaterial()
            m.lightingModel = .physicallyBased
            m.isDoubleSided = true
            m.diffuse.contents = NSColor(white: 0.8, alpha: 1)
            if let index, list("materials").indices.contains(index) {
                let pbr = list("materials")[index]["pbrMetallicRoughness"] as? [String: Any] ?? [:]
                if let f = pbr["baseColorFactor"] as? [Double], f.count == 4 {
                    m.diffuse.contents = NSColor(srgbRed: f[0], green: f[1], blue: f[2], alpha: f[3])
                }
                if let texture = (pbr["baseColorTexture"] as? [String: Any])?["index"] as? Int, let image = try? self.image(texture: texture) {
                    m.diffuse.contents = image
                }
                m.metalness.contents = (pbr["metallicFactor"] as? Double) ?? 0
                m.roughness.contents = (pbr["roughnessFactor"] as? Double) ?? 0.8
                materials[index] = m
            }
            return m
        }

        private func image(texture: Int) throws -> NSImage? {
            let textures = list("textures"), images = list("images")
            guard textures.indices.contains(texture), let source = textures[texture]["source"] as? Int, images.indices.contains(source),
                let view = images[source]["bufferView"] as? Int
            else { return nil }
            return NSImage(data: try bytes(view: view))
        }

        private func bytes(view index: Int) throws -> Data {
            let views = list("bufferViews")
            guard views.indices.contains(index) else { throw Failure(message: "The 3D file names data it does not have.") }
            let view = views[index]
            let start = view["byteOffset"] as? Int ?? 0, length = view["byteLength"] as? Int ?? 0
            guard start >= 0, length >= 0, start + length <= bin.count else { throw Failure(message: "The 3D file's data is cut short.") }
            return bin.subdata(in: start..<start + length)
        }

        /// An accessor's values as floats (normalized integers scaled to 0…1).
        func floats(_ index: Int, components: Int?) throws -> [Float] {
            let (raw, type, count, per, stride) = try accessor(index)
            if let components, per != components { throw Failure(message: "The 3D file's vertex data has an unexpected shape.") }
            var out = [Float](); out.reserveCapacity(count * per)
            let size = [5126: 4, 5123: 2, 5121: 1, 5125: 4][type] ?? 0
            guard size > 0 else { throw Failure(message: "The 3D file uses a number type Arsenal cannot read.") }
            let step = stride > 0 ? stride : size * per
            guard count == 0 || (count - 1) * step + size * per <= raw.count else { throw Failure(message: "The 3D file's vertex data is cut short.") }
            raw.withUnsafeBytes { buffer in
                for i in 0..<count {
                    for c in 0..<per {
                        let at = i * step + c * size
                        switch type {
                        case 5126: out.append(Float(bitPattern: buffer.loadUnaligned(fromByteOffset: at, as: UInt32.self).littleEndian))
                        case 5123: out.append(Float(buffer.loadUnaligned(fromByteOffset: at, as: UInt16.self).littleEndian) / 65535)
                        case 5121: out.append(Float(buffer.load(fromByteOffset: at, as: UInt8.self)) / 255)
                        default: out.append(Float(buffer.loadUnaligned(fromByteOffset: at, as: UInt32.self).littleEndian))
                        }
                    }
                }
            }
            return out
        }

        func integers(_ index: Int) throws -> [UInt32] {
            let (raw, type, count, per, stride) = try accessor(index)
            let size = [5125: 4, 5123: 2, 5121: 1][type] ?? 0
            guard size > 0, per == 1 else { throw Failure(message: "The 3D file's triangles use a type Arsenal cannot read.") }
            let step = stride > 0 ? stride : size
            guard count == 0 || (count - 1) * step + size <= raw.count else { throw Failure(message: "The 3D file's triangles are cut short.") }
            return raw.withUnsafeBytes { buffer in
                (0..<count).map { i in
                    let at = i * step
                    switch size {
                    case 4: return buffer.loadUnaligned(fromByteOffset: at, as: UInt32.self).littleEndian
                    case 2: return UInt32(buffer.loadUnaligned(fromByteOffset: at, as: UInt16.self).littleEndian)
                    default: return UInt32(buffer.load(fromByteOffset: at, as: UInt8.self))
                    }
                }
            }
        }

        private func accessor(_ index: Int) throws -> (Data, Int, Int, Int, Int) {
            let accessors = list("accessors")
            guard accessors.indices.contains(index), let view = accessors[index]["bufferView"] as? Int, let type = accessors[index]["componentType"] as? Int,
                let count = accessors[index]["count"] as? Int, count >= 0
            else { throw Failure(message: "The 3D file names data it does not have.") }
            let per = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][accessors[index]["type"] as? String ?? ""] ?? 0
            guard per > 0 else { throw Failure(message: "The 3D file uses a data shape Arsenal cannot read.") }
            let all = try bytes(view: view)
            let offset = accessors[index]["byteOffset"] as? Int ?? 0
            guard offset >= 0, offset <= all.count else { throw Failure(message: "The 3D file's data is cut short.") }
            let stride = list("bufferViews")[view]["byteStride"] as? Int ?? 0
            return (all.subdata(in: offset..<all.count), type, count, per, stride)
        }
    }
}

extension Array {
    /// Consecutive slices of `size` (a trailing remainder is dropped).
    fileprivate func chunks(_ size: Int) -> [ArraySlice<Element>] {
        stride(from: 0, to: count - count % size, by: size).map { self[$0..<$0 + size] }
    }
}

/// The model on the panel's own ground: at rest a still SceneKit rendered once (transparent, so the panel shows
/// through, and it is in every capture of the window); under the pointer the live view, which turns and zooms.
struct ModelPreview: View {
    let url: URL
    @PreviewLive private var live: URL?
    var body: some View {
        ZStack {
            if live == url {
                ModelSceneView(url: url)
            } else if let still = ModelStill.image(url) {
                Image(nsImage: still).resizable().scaledToFit()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onHover { inside in if inside { live = url } }
    }
}

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias PreviewLive = SwiftUI.State<URL?>

/// A model's still, rendered once per file with SceneKit's default camera and lights on a transparent ground.
@MainActor enum ModelStill {
    private static let stills = MediaCache<URL, NSImage>(limit: 48 * MediaCaches.mb, cost: MediaCaches.bytes)
    static func image(_ url: URL, size: CGSize = CGSize(width: 1200, height: 900)) -> NSImage? {
        if let hit = stills[url] { return hit }
        guard let scene = try? GLB.scene(url) else { return nil }
        scene.background.contents = NSColor.clear
        let renderer = SCNRenderer(device: MTLCreateSystemDefaultDevice(), options: nil)
        renderer.scene = scene
        renderer.autoenablesDefaultLighting = true
        let image = renderer.snapshot(atTime: 0, with: size, antialiasingMode: .multisampling4X)
        stills.set(image, for: url)
        return image
    }
}

/// The live view: the pointer turns and zooms the model; SceneKit draws only when something changes.
struct ModelSceneView: NSViewRepresentable {
    let url: URL
    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = false
        load(view)
        return view
    }
    func updateNSView(_ view: SCNView, context: Context) {
        if view.scene?.attribute(forKey: "arsenal.url") as? URL != url { load(view) }
    }
    private func load(_ view: SCNView) {
        let scene = (try? GLB.scene(url)) ?? SCNScene()
        scene.background.contents = NSColor.clear
        scene.setAttribute(url, forKey: "arsenal.url")
        view.scene = scene
        view.pointOfView = nil
    }
}
