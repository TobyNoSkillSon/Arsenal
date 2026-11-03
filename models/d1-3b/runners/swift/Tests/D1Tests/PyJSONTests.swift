// SPDX-License-Identifier: MIT
import D1Engine
import Foundation
import Testing

/// Python's json.dumps / str / repr as the vendor prompt applies them (expected strings from CPython 3.13).
@Test func floatRepr() {
    let cases: [(Double, String)] = [
        (12.5, "12.5"), (1e-05, "1e-05"), (0.0001, "0.0001"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (-0.0, "-0.0"),
        (1.5e300, "1.5e+300"), (100, "100.0"), (0.1, "0.1"), (123456789012345678, "1.2345678901234568e+17"), (5e-324, "5e-324")
    ]
    for (value, expected) in cases { #expect(PyJSON.floatRepr(value) == expected) }
}

@Test func dumpsIndent() throws {
    let v = try PyJSONParser.parse(Data(#"{"b":1,"a":[1.50,{"x":null,"y":true}],"e":{},"l":[],"s":"ż\"\n\u0001"}"#.utf8))
    #expect(
        v.dumps() == "{\n  \"b\": 1,\n  \"a\": [\n    1.5,\n    {\n      \"x\": null,\n      \"y\": true\n    }\n  ],\n  \"e\": {},\n  \"l\": [],\n  \"s\": \"ż\\\"\\n\\u0001\"\n}")
}

@Test func pythonStr() throws {
    let v = try PyJSONParser.parse(Data(#"{"note":"it's","q":"a'b\"c","n":[1,2.0,null,false],"t":"tab\there "}"#.utf8))
    #expect(v.pyStr() == #"{'note': "it's", 'q': 'a\'b"c', 'n': [1, 2.0, None, False], 't': 'tab\there\xa0'}"#)
    #expect(PyJSON.string("plain").pyStr() == "plain")
}

@Test func duplicateKeysKeepFirstPlaceLastValue() throws {
    let v = try PyJSONParser.parse(Data(#"{"a":1,"b":2,"a":3}"#.utf8))
    #expect(v.compact() == #"{"a":3,"b":2}"#)
}

@Test func truthiness() throws {
    for (text, truthy) in [("0", false), ("0.0", false), ("-0", false), ("1e-9", true), ("\"\"", false), ("[]", false), ("{}", false), ("null", false), ("\"x\"", true)] {
        #expect(try PyJSONParser.parse(Data(text.utf8)).truthy == truthy, "\(text)")
    }
}
