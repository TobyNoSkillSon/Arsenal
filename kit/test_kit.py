#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kit checks with no model: the template runner builds, says hello as Arsenal expects, casts weights through a symlinked
folder (Get's staging path), and loads a toy checkpoint. Run: python3 kit/test_kit.py (first run compiles MLX)."""
import json, os, shutil, struct, subprocess, sys, tempfile, unittest

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
kit = __import__("new-model")
ID = "kit-selftest"


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, b""
    for name, (shape, values) in tensors.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs += data
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + blobs)


def write_torch_checkpoint(path, state, nested_under=None):
    """A torch.save ZIP archive written with the standard library only: stored entries, a protocol-2 data.pkl whose
    globals are the names torch writes (stand-in modules exist only while pickling), one storage per tensor.
    `state` maps a name to (shape, dtype, raw little-endian bytes); dtype is "BFloat16", "Float" or "Long"."""
    import collections, io, pickle, sys, types, zipfile
    fake_torch, fake_utils = types.ModuleType("torch"), types.ModuleType("torch._utils")
    storages = {}
    for kind in ("BFloat16", "Float", "Long", "Double"):
        storages[kind] = type(kind + "Storage", (), {"__module__": "torch"})
        setattr(fake_torch, kind + "Storage", storages[kind])

    def _rebuild_tensor_v2(*args):
        raise AssertionError("never called")
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    fake_utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    class Storage:
        def __init__(self, key, kind, numel):
            self.key, self.kind, self.numel = key, kind, numel

    class Tensor:
        def __init__(self, storage, shape, stride=None):
            self.storage, self.shape, self.stride = storage, shape, stride

        def __reduce__(self):
            stride, step = [], 1
            for dim in reversed(self.shape):
                stride.insert(0, step)
                step *= dim
            stride = self.stride or stride
            return (_rebuild_tensor_v2, (self.storage, 0, tuple(self.shape), tuple(stride), False, collections.OrderedDict()))

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", storages[obj.kind], obj.key, "cpu", obj.numel)
            return None

    width = {"BFloat16": 2, "Float": 4, "Long": 8, "Double": 8}
    tensors, blobs = collections.OrderedDict(), {}
    for index, (name, (shape, kind, data, *stride)) in enumerate(state.items()):
        tensors[name] = Tensor(Storage(str(index), kind, len(data) // width[kind]), shape, stride[0] if stride else None)
        blobs[str(index)] = data
    root = collections.OrderedDict([(nested_under, tensors), ("epoch", 3)]) if nested_under else tensors
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    sys.modules["torch"], sys.modules["torch._utils"] = fake_torch, fake_utils
    try:
        buffer = io.BytesIO()
        Pickler(buffer, protocol=2).dump(root)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        for key, data in blobs.items():
            archive.writestr("archive/data/" + key, data)


def bf16(values):
    return b"".join(struct.pack("<f", v)[2:] for v in values)


def write_npy(path_or_none, array_shape, values, descr="<f8"):
    header = "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, "".join(f"{d}," for d in array_shape))
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + struct.pack(f"<{len(values)}d", *values)
    if path_or_none:
        open(path_or_none, "wb").write(data)
    return data


def read_safetensors(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        body = handle.read()
    return header, body


def write_raw_archive(path, pickle_bytes, storages=None):
    """A stored ZIP holding exactly this data.pkl (and optional storages): for hostile-input tests."""
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", pickle_bytes)
        for key, data in (storages or {}).items():
            archive.writestr("archive/data/" + key, data)


def zip64_entry_without_its_fields(path):
    """A ZIP whose one central entry says its sizes are in a ZIP64 extra field, then gives that field no room."""
    name = b"archive/data.pkl"
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 45, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, len(name), 0) + name
    extra = struct.pack("<HH", 0x0001, 0)
    central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 45, 45, 0, 0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, len(name), len(extra),
                          0, 0, 0, 0, 0) + name + extra
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
    open(path, "wb").write(local + central + end)


def read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))


def exchange(runner, requests):
    process = subprocess.Popen([runner], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for message in requests:
        kit.send(process, message)
    replies = []
    for frame in kit.frames(process):
        replies.append(frame)
        if frame.get("id") == requests[-1]["id"] and "ok" in frame:
            break
    process.kill()
    return replies


class KitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target = kit.folder(ID)
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(os.path.join(target, "runners/swift"))
        shutil.copy(os.path.join(KIT, "runner/Package.swift"), os.path.join(target, "runners/swift/Package.swift"))
        shutil.copytree(os.path.join(KIT, "runner/Sources"), os.path.join(target, "runners/swift/Sources"))
        # The test copy gains one op, "big", that returns `count` numbers, to drive Wire's reply-size guard.
        model = os.path.join(target, "runners/swift/Sources/Runner/Model.swift")
        text = open(model).read()
        text = text.replace('let modelOps = ["predict"]', 'let modelOps = ["predict", "big"]')
        text = text.replace("        let values = try params.numbers(\"values\")\n",
                            "        if op == \"big\" { return [\"values\": Array(repeating: 0.123456789012, count: params.raw[\"count\"] as? Int ?? 0)] }\n"
                            "        let values = try params.numbers(\"values\")\n", 1)
        assert '"big"' in text and 'op == "big"' in text, "the template Model.swift changed; update the big-op patch"
        open(model, "w").write(text)
        kit.build(type("Args", (), {"id": ID})())
        cls.runner = os.path.join(target, "runners/swift/bin/runner")
        cls.scratch = tempfile.mkdtemp(dir="/tmp")  # /tmp is a symlink to /private/tmp, as Get's paths can be

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(kit.folder(ID), ignore_errors=True)
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def test_receipt_covers_every_runner_file(self):
        receipt = json.load(open(os.path.join(kit.folder(ID), "runners/swift/build-receipt.json")))
        self.assertIn("runners/swift/bin/runner", receipt["sha256"])
        self.assertIn("runners/swift/Package.resolved", receipt["sha256"])
        self.assertTrue(any(k.endswith("default.metallib") for k in receipt["sha256"]))

    def test_hello_names_the_kit_runner(self):
        hello = exchange(self.runner, [{"id": 1, "op": "ping"}])[0]
        self.assertEqual((hello["event"], hello["runner"], hello["proto"]), ("hello", "kit", 1))
        self.assertIn("local.example", hello["kinds"])
        self.assertIn("convert", hello["ops"])

    def test_convert_keeps_relative_paths_through_a_symlinked_folder(self):
        source, destination = os.path.join(self.scratch, "src"), os.path.join(self.scratch, "dst")
        os.makedirs(os.path.join(source, "1_Pooling"))
        os.makedirs(destination)
        write_safetensors(os.path.join(source, "model.safetensors"), {"w": ([2, 2], [1.0, 2.0, 3.0, 4.0])})
        open(os.path.join(source, "1_Pooling/config.json"), "w").write("{}")
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(sorted(f["path"] for f in reply["files"]), ["1_Pooling/config.json", "model.safetensors"])
        self.assertEqual(read_header(os.path.join(destination, "model.safetensors"))["w"]["dtype"], "F16")

    def test_load_and_run_the_template_model(self):
        weights = os.path.join(self.scratch, "weights")
        os.makedirs(weights, exist_ok=True)
        json.dump({"input_size": 4, "num_labels": 2}, open(os.path.join(weights, "config.json"), "w"))
        write_safetensors(os.path.join(weights, "model.safetensors"),
                          {"classifier.weight": ([2, 4], [1, 0, 0, 0, 0, 1, 0, 0]), "classifier.bias": ([2], [0, 0])})
        replies = exchange(self.runner, [
            {"id": 1, "op": "load", "model_dir": kit.folder(ID), "weights_dir": weights, "dtype": "fp16", "config": {"settings": {}}},
            {"id": 2, "op": "predict", "values": [2, 0, 0, 0]}])
        result = replies[-1]
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(sum(result["probabilities"]), 1, places=3)
        self.assertGreater(result["probabilities"][0], result["probabilities"][1])

    def test_unknown_op_is_a_framed_error(self):
        reply = exchange(self.runner, [{"id": 1, "op": "nonsense"}])[-1]
        self.assertEqual(reply["error"]["code"], "unsupported_op")

    def test_nan_output_is_an_error_not_an_answer(self):
        weights = os.path.join(self.scratch, "weights")
        os.makedirs(weights, exist_ok=True)
        json.dump({"input_size": 4, "num_labels": 2}, open(os.path.join(weights, "config.json"), "w"))
        write_safetensors(os.path.join(weights, "model.safetensors"),
                          {"classifier.weight": ([2, 4], [1, 0, 0, 0, 0, 1, 0, 0]), "classifier.bias": ([2], [0, 0])})
        replies = exchange(self.runner, [
            {"id": 1, "op": "load", "model_dir": kit.folder(ID), "weights_dir": weights, "dtype": "fp16", "config": {"settings": {}}},
            {"id": 2, "op": "predict", "values": [1e30, -1e30, 0, 0]}])
        self.assertFalse(replies[-1]["ok"])
        self.assertIn("NaN or infinite", replies[-1]["error"]["message"])

    def _loaded(self, *requests):
        weights = os.path.join(self.scratch, "weights")
        os.makedirs(weights, exist_ok=True)
        json.dump({"input_size": 4, "num_labels": 2}, open(os.path.join(weights, "config.json"), "w"))
        write_safetensors(os.path.join(weights, "model.safetensors"),
                          {"classifier.weight": ([2, 4], [1, 0, 0, 0, 0, 1, 0, 0]), "classifier.bias": ([2], [0, 0])})
        return exchange(self.runner, [
            {"id": 1, "op": "load", "model_dir": kit.folder(ID), "weights_dir": weights, "dtype": "fp16", "config": {"settings": {}}},
            *requests])

    def test_a_reply_over_the_frame_cap_is_a_422_and_the_runner_lives(self):
        # About 14 bytes a number: 1.3 million numbers is ~18 MB, over Arsenal's 16 MiB frame cap.
        replies = self._loaded({"id": 2, "op": "big", "count": 1_300_000}, {"id": 3, "op": "big", "count": 1000})
        refused = next(r for r in replies if r.get("id") == 2 and "ok" in r)
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["error"]["code"], "invalid_request")
        self.assertRegex(refused["error"]["message"], r"reply would be \d+\.\d MiB, over the 16 MiB limit")
        self.assertTrue(replies[-1]["ok"], replies[-1])
        self.assertEqual(len(replies[-1]["values"]), 1000)

    def test_a_reply_just_under_the_frame_cap_is_sent(self):
        # Each number is "0.123456789012," (15 bytes); the envelope is measured on a small reply. Counts a few KB either
        # side of Wire.replyLimit (16 MiB - 64 KiB, body plus type byte): under goes out, over is refused.
        small = self._loaded({"id": 2, "op": "big", "count": 1000})[-1]
        envelope = len(json.dumps(small, separators=(",", ":"))) - 15 * 1000 + 1
        limit = 16 * 1024 * 1024 - 64 * 1024
        under = (limit - 1 - envelope - 3000) // 15
        replies = self._loaded({"id": 2, "op": "big", "count": under}, {"id": 3, "op": "big", "count": under + 400})
        sent = next(r for r in replies if r.get("id") == 2 and "ok" in r)
        self.assertTrue(sent["ok"], sent.get("error"))
        self.assertEqual(len(sent["values"]), under)
        refused = next(r for r in replies if r.get("id") == 3 and "ok" in r)
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["error"]["code"], "invalid_request")

    def test_convert_keeps_tensors_beyond_half_range_in_float32(self):
        source, destination = os.path.join(self.scratch, "big-src"), os.path.join(self.scratch, "big-dst")
        os.makedirs(source)
        os.makedirs(destination)
        write_safetensors(os.path.join(source, "model.safetensors"), {"var": ([2], [1.0, 1e6]), "w": ([2], [1.0, 2.0])})
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertTrue(reply["ok"], reply)
        header = read_header(os.path.join(destination, "model.safetensors"))
        self.assertEqual((header["var"]["dtype"], header["w"]["dtype"]), ("F32", "F16"))

    def test_convert_turns_a_torch_archive_into_safetensors_without_running_its_pickle(self):
        source, destination = os.path.join(self.scratch, "pth-src"), os.path.join(self.scratch, "pth-dst")
        os.makedirs(os.path.join(source, "data/weights"))
        os.makedirs(destination)
        write_torch_checkpoint(os.path.join(source, "data/weights/flat.pth"), {
            "blocks.0.w": ([2, 2], "BFloat16", bf16([1.0, -2.0, 0.5, 3.0])),
            "blocks.0.big": ([2], "Float", struct.pack("<2f", 1.0, 1e6)),
            "steps": ([1], "Long", struct.pack("<q", 7))})
        write_torch_checkpoint(os.path.join(source, "nested.pt"), {"head.b": ([3], "BFloat16", bf16([0.25, 0.5, 1.0]))},
                               nested_under="model")
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(sorted(f["path"] for f in reply["files"]), ["data/weights/flat.safetensors", "nested.safetensors"])
        self.assertFalse(os.path.exists(os.path.join(destination, "data/weights/flat.pth")))
        header, body = read_safetensors(os.path.join(destination, "data/weights/flat.safetensors"))
        self.assertEqual(sorted(k for k in header if k != "__metadata__"), ["blocks.0.big", "blocks.0.w", "steps"])
        self.assertEqual((header["blocks.0.w"]["dtype"], header["blocks.0.w"]["shape"]), ("F16", [2, 2]))
        self.assertEqual(header["blocks.0.big"]["dtype"], "F32")  # beyond FP16's range
        self.assertEqual(header["steps"]["dtype"], "I64")
        start, end = header["blocks.0.w"]["data_offsets"]
        self.assertEqual(list(struct.unpack("<4e", body[start:end])), [1.0, -2.0, 0.5, 3.0])
        start, end = header["steps"]["data_offsets"]
        self.assertEqual(struct.unpack("<q", body[start:end])[0], 7)
        nested, body = read_safetensors(os.path.join(destination, "nested.safetensors"))
        self.assertEqual(sorted(k for k in nested if k != "__metadata__"), ["model.head.b"])
        start, end = nested["model.head.b"]["data_offsets"]
        self.assertEqual(list(struct.unpack("<3e", body[start:end])), [0.25, 0.5, 1.0])

    def test_convert_refuses_a_pickle_that_needs_code(self):
        import pickle, zipfile
        source, destination = os.path.join(self.scratch, "evil-src"), os.path.join(self.scratch, "evil-dst")
        os.makedirs(source)
        os.makedirs(destination)
        with zipfile.ZipFile(os.path.join(source, "evil.pth"), "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", pickle.dumps({"w": os.getcwd}, protocol=2))  # a global call, never made
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertFalse(reply["ok"])
        self.assertIn("no tensors", reply["error"]["message"])

    def test_convert_stores_numpy_arrays_losslessly(self):
        import zipfile
        source, destination = os.path.join(self.scratch, "np-src"), os.path.join(self.scratch, "np-dst")
        os.makedirs(source)
        os.makedirs(destination)
        write_npy(os.path.join(source, "idf.npy"), [3], [0.1, 2.5, 1e300])
        with zipfile.ZipFile(os.path.join(source, "planes.npz"), "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("0.npy", write_npy(None, [2, 2], [1.0, -1.0, 0.5, 0.25]))
            archive.writestr("1.npy", write_npy(None, [1, 2], [3.0, 4.0]))
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertTrue(reply["ok"], reply)
        self.assertEqual(sorted(f["path"] for f in reply["files"]), ["idf.safetensors", "planes.safetensors"])
        header, body = read_safetensors(os.path.join(destination, "idf.safetensors"))
        self.assertEqual((header["array"]["dtype"], header["array"]["shape"]), ("F64", [3]))
        start, end = header["array"]["data_offsets"]
        self.assertEqual(list(struct.unpack("<3d", body[start:end])), [0.1, 2.5, 1e300])
        header, body = read_safetensors(os.path.join(destination, "planes.safetensors"))
        self.assertEqual({k: header[k]["shape"] for k in ("0", "1")}, {"0": [2, 2], "1": [1, 2]})
        start, end = header["1"]["data_offsets"]
        self.assertEqual(list(struct.unpack("<2d", body[start:end])), [3.0, 4.0])

    def convert_reply(self, name, write):
        """Convert one crafted file; the runner must answer (not crash or hang) within a few seconds."""
        import time
        source, destination = os.path.join(self.scratch, name + "-src"), os.path.join(self.scratch, name + "-dst")
        os.makedirs(source)
        os.makedirs(destination)
        write(source)
        started = time.time()
        reply = exchange(self.runner, [{"id": 1, "op": "convert", "src_dir": source, "dst_dir": destination, "dtype": "fp16",
                                        "group_size": 64, "recipe": "cast:arsenal-kit-half-v1", "assets": {}}])[-1]
        self.assertLess(time.time() - started, 20, f"{name} took too long")
        self.assertIn("ok", reply, f"{name}: the runner did not answer")
        return reply

    def assert_refused(self, name, write, words):
        reply = self.convert_reply(name, write)
        self.assertFalse(reply["ok"], f"{name} was accepted")
        self.assertIn(words, reply["error"]["message"])

    def test_hostile_shared_tuples_are_refused_in_linear_time(self):
        # Review item 1: a tuple of the same memoised tuple, 80 levels deep, used to cost about 3^33 depth checks.
        ops = b"\x80\x02)q\x00"  # PROTO 2, EMPTY_TUPLE, BINPUT 0
        for _ in range(80):
            ops += b"h\x00h\x00h\x00\x87q\x00"  # BINGET 0 x3, TUPLE3, BINPUT 0
        ops += b"."
        self.assert_refused("tuples", lambda d: write_raw_archive(os.path.join(d, "t.pth"), ops), "nesting too deep")

    def test_hostile_dictionary_keys_must_be_strings_or_integers(self):
        # Review item 1: a tuple key used to be compared against every key so far.
        ops = b"\x80\x02}q\x00)K\x01s."  # EMPTY_DICT, BINPUT, EMPTY_TUPLE (key), BININT1 1, SETITEM, STOP
        self.assert_refused("keys", lambda d: write_raw_archive(os.path.join(d, "k.pth"), ops), "not a string or an integer")

    def test_hostile_nested_lists_are_refused(self):
        # Review item 2: 100,000 nested lists used to overflow the stack when released.
        ops = b"\x80\x02" + b"]" * 100_000 + b"a" * 99_999 + b"."
        self.assert_refused("lists", lambda d: write_raw_archive(os.path.join(d, "l.pth"), ops), "nesting too deep")

    def test_hostile_container_changed_after_nesting_is_refused(self):
        # A list appended into itself (a cycle) or filled after it was nested would make the depth wrong.
        ops = b"\x80\x02]q\x00h\x00a."  # EMPTY_LIST, BINPUT, BINGET (the same list), APPEND into itself
        self.assert_refused("cycle", lambda d: write_raw_archive(os.path.join(d, "c.pth"), ops), "after it was nested")

    def test_hostile_zip64_extra_field_without_room_is_refused(self):
        # Review item 3: the reader used to read past the extra field.
        self.assert_refused("zip64", lambda d: zip64_entry_without_its_fields(os.path.join(d, "z.pth")), "ZIP64 extra field")

    def test_hostile_npy_shape_overflow_is_refused(self):
        # Review item 4: count x width used to overflow and trap.
        def write(d):
            header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2305843009213693952,), }"
            header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
            open(os.path.join(d, "big.npy"), "wb").write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode())
        self.assert_refused("npy", write, "does not match")

    def test_hostile_contiguity_overflow_is_refused(self):
        # Review item 4: an empty dimension hid an overflowing product of the others.
        def write(d):
            write_torch_checkpoint(os.path.join(d, "s.pth"), {"w": ([0, 1 << 40, 1 << 40], "Float", b"", [0, 1 << 40, 1])})
        self.assert_refused("strides", write, "Invalid tensor shape")

    def test_hostile_memo_flood_is_refused(self):
        # Review item 5: memo writes were capped only when a value was pushed.
        ops = b"\x80\x02N" + b"".join(b"r" + struct.pack("<I", n) for n in range(1_000_100)) + b"."
        self.assert_refused("memo", lambda d: write_raw_archive(os.path.join(d, "m.pth"), ops), "too many memo entries")

    def test_float64_storage_is_refused_before_mlx(self):
        # Review item 13: float64 never reaches MLX.
        reply = self.convert_reply("f64", lambda d: write_torch_checkpoint(os.path.join(d, "d.pth"), {"w": ([2], "Double", struct.pack("<2d", 1, 2))}))
        self.assertFalse(reply["ok"])
        self.assertIn("float64", reply["error"]["message"])

    def test_gpu_selftest_matches_the_cpu(self):
        reply = exchange(self.runner, [{"id": 1, "op": "selftest"}])[-1]
        self.assertTrue(reply["ok"], reply)
        self.assertLess(reply["max_relative_difference"], 1e-2)

    def test_a_failed_selftest_leaves_no_runner_for_try_to_reuse(self):
        target = kit.folder(ID)
        original = kit.gpu_selftest
        kit.gpu_selftest = lambda runner: kit.die("simulated self-test failure")
        try:
            with self.assertRaises(SystemExit):
                kit.build(type("Args", (), {"id": ID})())
            self.assertTrue(kit.needs_build(target))
        finally:
            kit.gpu_selftest = original
            kit.build(type("Args", (), {"id": ID})())


class PinTests(unittest.TestCase):
    def test_kit_package_pins_the_same_mlx_as_the_kit_check(self):
        self.assertIn(f'revision: "{kit.MLX_PIN}"', open(os.path.join(KIT, "runner/Package.swift")).read())

    def test_installed_app_records_the_kit_mlx_pin(self):
        self.assertEqual(kit.app_mlx_pin(), kit.MLX_PIN)

    def test_weights_live_where_arsenal_get_looks(self):
        # ArsenalCore LocalModelSources: ~/.cache/huggingface/hub/models--owner--repo/snapshots/<revision>/<file>
        self.assertEqual(kit.snapshot_dir("o/r", "a" * 40),
                         os.path.expanduser("~/.cache/huggingface/hub/models--o--r/snapshots/" + "a" * 40))


    def test_a_snapshot_linked_into_another_apps_folder_is_never_written(self):
        # A download of the Fish Audio pins once wrote through such a link over another app's Drama 3 weights.
        root = tempfile.mkdtemp(dir="/tmp")
        saved = kit.HF_HUB
        try:
            kit.HF_HUB = os.path.join(root, "hub")
            foreign = os.path.join(root, "other-app", "voice-model")
            os.makedirs(foreign)
            open(os.path.join(foreign, "model-00001.safetensors"), "w").write("theirs")
            snapshots = os.path.join(kit.HF_HUB, "models--o--r", "snapshots")
            os.makedirs(snapshots)
            os.symlink(foreign, os.path.join(snapshots, "a" * 40))
            with self.assertRaises(SystemExit):
                kit.writable_snapshot_dir("o/r", "a" * 40)
            # A linked repo folder: nothing, not even an empty snapshots folder, is created behind the link.
            os.symlink(foreign, os.path.join(kit.HF_HUB, "models--o--linked"))
            with self.assertRaises(SystemExit):
                kit.writable_snapshot_dir("o/linked", "a" * 40)
            # A dangling snapshot link stops cleanly too.
            os.symlink(os.path.join(root, "gone"), os.path.join(snapshots, "c" * 40))
            with self.assertRaises(SystemExit):
                kit.writable_snapshot_dir("o/r", "c" * 40)
            self.assertFalse(os.path.exists(os.path.join(root, "gone")))
            self.assertEqual(os.listdir(foreign), ["model-00001.safetensors"])
            self.assertEqual(open(os.path.join(foreign, "model-00001.safetensors")).read(), "theirs")
            own = kit.writable_snapshot_dir("o/s", "b" * 40)
            self.assertTrue(kit.inside_hub(own))
            self.assertFalse(kit.inside_hub(foreign))
        finally:
            kit.HF_HUB = saved
            shutil.rmtree(root)


class OfflineTests(unittest.TestCase):
    def test_schema_errors_name_the_missing_and_unknown_fields(self):
        schema = {"type": "object", "additionalProperties": False, "required": ["embeddings"],
                  "properties": {"embeddings": {"type": "array", "items": {"type": "array"}}}}
        self.assertEqual(kit.schema_errors({"embeddings": [[0.1]]}, schema), [])
        errors = kit.schema_errors({"vectors": [1]}, schema)
        self.assertIn("result.embeddings is required", errors)
        self.assertIn("result.vectors is not in the schema", errors)
        self.assertEqual(kit.schema_errors({"embeddings": [1]}, schema), ["result.embeddings[0] must be array, got int"])

    def test_inventory_keeps_one_checkpoint_and_always_pins_tokenizers(self):
        info = {"id": "o/r", "sha": "0" * 40, "siblings": [
            {"rfilename": "model.safetensors", "size": 10}, {"rfilename": "model.fp16.safetensors", "size": 5},
            {"rfilename": "tokenizer.json", "size": 40_000_000}, {"rfilename": "config.json", "size": 1},
            {"rfilename": "adapter_model.safetensors", "size": 1}, {"rfilename": "README.md", "size": 1}]}
        weights, small, _ = kit.inventory(info)
        self.assertEqual([w["rfilename"] for w in weights], ["model.safetensors"])
        self.assertEqual(sorted(s["rfilename"] for s in small), ["config.json", "tokenizer.json"])

    def test_outline_lists_classes_and_methods_with_lines(self):
        folder = tempfile.mkdtemp(dir="/tmp")
        open(os.path.join(folder, "transformers-modeling_x.py"), "w").write("class Block(nn.Module):\n    def forward(self, x):\n        return x\n")
        kit.write_outline(folder)
        outline = open(os.path.join(folder, "outline.txt")).read()
        self.assertIn("1  class Block(nn.Module)", outline)
        self.assertIn("2      def forward(self, x)", outline)
        shutil.rmtree(folder)

    def test_tensor_tree_folds_layer_numbers(self):
        folder = tempfile.mkdtemp(dir="/tmp")
        lines = [f"layers.{i}.attn.weight [4, 4] F32" for i in range(12)] + ["head.weight [2, 4] F32"]
        kit.write_tensor_tree(folder, lines)
        tree = open(os.path.join(folder, "tensors-tree.txt")).read().splitlines()
        self.assertEqual(tree, ["head.weight [2, 4] F32", "layers.{0-11}.attn.weight [4, 4] F32"])
        shutil.rmtree(folder)

    def test_tensor_tree_folds_every_position(self):
        folder = tempfile.mkdtemp(dir="/tmp")
        lines = [f"down.{i}.block.{j}.conv.weight [3] F32" for i in range(4) for j in range(2)] + ["a.0.1.b [1] F32", "a.0.2.b [1] F32"]
        kit.write_tensor_tree(folder, lines)
        tree = open(os.path.join(folder, "tensors-tree.txt")).read().splitlines()
        self.assertEqual(tree, ["a.0.{1-2}.b [1] F32", "down.{0-3}.block.{0-1}.conv.weight [3] F32"])
        shutil.rmtree(folder)

    def test_try_clears_a_half_converted_folder_before_converting(self):
        # Review item 11: leftovers of an interrupted conversion made the next convert refuse ("both convert to").
        root = tempfile.mkdtemp(dir="/tmp")
        saved = (kit.WORK, kit.HF_HUB, kit.send, os.environ.pop("ARSENAL_KIT_CONVERTED", None))
        try:
            kit.WORK, kit.HF_HUB = os.path.join(root, "work"), os.path.join(root, "hub")
            folder = kit.folder("toy")
            os.makedirs(folder)
            open(os.path.join(folder, "model.toml"), "w").write(
                'id = "toy"\n[weights]\nrepo = "o/toy"\nrevision = "' + "a" * 40 + '"\nfiles = ["idf.npy"]\n')
            snapshot = kit.snapshot_dir("o/toy", "a" * 40)
            os.makedirs(snapshot)
            write_npy(os.path.join(snapshot, "idf.npy"), [1], [1.0])
            target = os.path.join(kit.WORK, "converted", "toy")
            os.makedirs(target)
            open(os.path.join(target, "idf.safetensors"), "w").write("left over")
            seen = []
            kit.send = lambda process, message: seen.append((message, sorted(os.listdir(target))))
            result = kit.load_dir("toy", None, lambda request_id: {"ok": True, "files": []})
            self.assertEqual(result, target)
            self.assertEqual(seen[0][0]["op"], "convert")
            self.assertEqual(seen[0][1], [], "the target was emptied before convert ran")
            self.assertTrue(os.path.exists(os.path.join(target, ".converted")))
            # A folder the user named is never deleted: with files and no marker it is refused.
            named = os.path.join(root, "mine")
            os.makedirs(named)
            open(os.path.join(named, "keep.txt"), "w").write("mine")
            os.environ["ARSENAL_KIT_CONVERTED"] = named
            with self.assertRaises(SystemExit):
                kit.load_dir("toy", None, lambda request_id: {"ok": True, "files": []})
            self.assertTrue(os.path.exists(os.path.join(named, "keep.txt")))
        finally:
            kit.WORK, kit.HF_HUB, kit.send = saved[:3]
            os.environ.pop("ARSENAL_KIT_CONVERTED", None)
            if saved[3] is not None:
                os.environ["ARSENAL_KIT_CONVERTED"] = saved[3]
            shutil.rmtree(root, ignore_errors=True)

    def test_licence_list_with_one_entry_is_read(self):
        self.assertEqual(kit.licence_of({"cardData": {"license": ["MIT"]}}), "mit")


if __name__ == "__main__":
    unittest.main()
