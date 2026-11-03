// SPDX-License-Identifier: MIT
import Foundation
import Security
import Darwin

extension APIRouter {
    func prepareMCP() throws {
        pruneMCPInbox()
        loadMCPJobRecords()
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let path = support.appendingPathComponent("mcp-token").path
        let descriptor = Darwin.open(path, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw HTTPFailure(500, "Cannot open MCP token file safely.") }
        defer { Darwin.close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(), fchmod(descriptor, 0o600) == 0 else {
            throw HTTPFailure(500, "MCP token must be an owner-only regular file.")
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        if info.st_size == 0 {
            var bytes = [UInt8](repeating: 0, count: 32)
            guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else { throw HTTPFailure(500, "Cannot create MCP token.") }
            let token = bytes.map { String(format: "%02x", $0) }.joined()
            try handle.write(contentsOf: Data((token + "\n").utf8)); try handle.synchronize(); mcpToken = token
        } else {
            guard info.st_size <= 256, let data = try handle.readToEnd(), let token = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                token.count == 64, token.allSatisfy({ $0.isHexDigit })
            else { throw HTTPFailure(500, "MCP token file is invalid; restore a valid token before starting Arsenal.") }
            mcpToken = token
        }
    }
    func authorizeMCP(_ request: HTTPRequest) throws {
        guard request.headers["origin"] == nil else { throw HTTPFailure(403, "Browser origins are not accepted.") }
        guard ["127.0.0.1:\(port)", "localhost:\(port)"].contains(request.headers["host"]?.lowercased() ?? "") else {
            throw HTTPFailure(403, "MCP Host must match the loopback listener.")
        }
        guard let mcpToken else { throw HTTPFailure(503, "MCP token is not initialized.") }
        let supplied = Array((request.headers["authorization"] ?? "").utf8), expected = Array(("Bearer " + mcpToken).utf8)
        var difference = supplied.count ^ expected.count
        for index in expected.indices { difference |= Int(expected[index] ^ (index < supplied.count ? supplied[index] : 0)) }
        guard difference == 0 else { throw HTTPFailure(401, "Send Authorization: Bearer <token from support/mcp-token>.") }
    }
}
