// SPDX-License-Identifier: MIT
import Foundation
import Testing
import D1Engine
@testable import D1Worker

private final class PrecisionDecisionFixture: DecisionEngine {
    var convKernelOK: Bool { true }
    func loadPictures(_ urls: [URL], budget: inout PictureBudget) throws -> [Picture] { [] }
    func tokens(_ state: PyJSON?, _ questions: [D1Engine.Question]) throws -> Int { 17 }
    func imageTokens(_ state: PyJSON?, _ questions: [D1Engine.Question], _ pictures: [Picture]) throws -> Int { 17 }
    func run(_ requests: [D1Engine.Request], cancelled: () -> Bool) throws -> [Outcome] {
        requests.map { request in .success((probabilities: request.questions.map { _ in [0.8, 0.2] }, read: 17)) }
    }
}

@Test(arguments: ["bf16", "int8", "int4"]) func d1ReportsTheLoadedPrecisionInDecideAndJudge(dtype: String) throws {
    let bits = dtype == "int4" ? 4 : dtype == "int8" ? 8 : 16
    let runner = Backend(engine: PrecisionDecisionFixture(), precision: dtype)
    let body = try PyJSONParser.parse(Data("{\"state\":\"A refund request\",\"questions\":{\"refund\":{\"type\":\"noul\",\"instructions\":\"Refund?\"}},\"bits\":\(bits)}".utf8))
    let fields = try runner.decide(body, cancelled: { false })
    #expect(fields.first { $0.0 == "bits" }?.1.int == bits)
    let judge = try PyJSONParser.parse(Data("{\"items\":[\"A refund request\"],\"questions\":{\"refund\":{\"type\":\"noul\",\"instructions\":\"Refund?\"}},\"bits\":\(bits)}".utf8))
    let batched = try runner.judge(judge, cancelled: { false })
    #expect(batched.first { $0.0 == "bits" }?.1.int == bits)
}
