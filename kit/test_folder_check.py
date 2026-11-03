#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Static folder checks and kit input handling, with no model, network or build. Run: python3 kit/test_folder_check.py"""
import io, json, os, re, shutil, struct, sys, tempfile, unittest, urllib.error
from unittest import mock

KIT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(KIT)
sys.path.insert(0, KIT)
import folder_check  # noqa: E402
kit = __import__("new-model")

MODEL_SWIFT = '''import Foundation
import MLX
import MLXNN
let modelKind = "local.labels"
let modelOps = ["classify"]
final class Model {}
'''


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, b""
    for name, shape in tensors.items():
        count = 1
        for d in shape:
            count *= d
        data = struct.pack(f"<{count}f", *([0.5] * count))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs += data
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + blobs)


class Fixture:
    """A complete, valid local folder plus its Hugging Face cache snapshot, all in a temporary directory."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="arsenal-check-")
        self.hub = os.path.join(self.root, "hub")
        self.folder = os.path.join(self.root, "toy-labels")
        revision = "a" * 40
        snapshot = os.path.join(self.hub, "models--owner--toy", "snapshots", revision)
        os.makedirs(snapshot)
        weights = os.path.join(snapshot, "model.safetensors")
        write_safetensors(weights, {"classifier.weight": [3, 4], "classifier.bias": [3]})
        open(os.path.join(snapshot, "config.json"), "w").write("{}")
        hashes = {n: folder_check.sha256(os.path.join(snapshot, n)) for n in ("model.safetensors", "config.json")}
        sizes = {n: os.path.getsize(os.path.join(snapshot, n)) for n in ("model.safetensors", "config.json")}
        for path in ("ops", "log", "results/quick-checks", "runners/swift/Sources/Runner", "runners/swift/bin/mlx-swift_Cmlx.bundle"):
            os.makedirs(os.path.join(self.folder, path))
        self.toml = os.path.join(self.folder, "model.toml")
        open(self.toml, "w").write(f'''schema = 3
id = "toy-labels"
name = "Toy Labels 1.0"
kind = "local.labels"
kind_file = "kinds/local.labels.toml"
summary = "Labels four numbers."
description = "A toy."
purpose = "Label numbers"
publisher = "owner"
released = 2026
params = "15"
tasks = ["text-classification"]
languages = ["mul"]  # no language
good_for = ["tests"]
limits = ["a toy, with a # inside a string"]
suggested = false

[licence]
spdx = "MIT"
name = "MIT"
commercial = true

[facets]
modalities_in = ["text"]
modalities_out = ["text"]

[weights]
repo = "owner/toy"
revision = "{revision}"
dtype = "fp32"
files = ["model.safetensors", "config.json"]

[weights.sha256]
"model.safetensors" = "{hashes['model.safetensors']}"
"config.json" = "{hashes['config.json']}"

[weights.bytes]
"model.safetensors" = {sizes['model.safetensors']}
"config.json" = {sizes['config.json']}

[[runner]]
name = "kit"
language = "swift"
entry = "runners/swift/bin/runner"
protocol = 1
chips = ["*"]

[[op]]
name = "classify"
title = "Classify"
input = "ops/classify.input.json"
output = "json"
output_schema = "ops/classify.output.json"
''')
        json.dump({"type": "object", "required": ["values"], "properties": {"values": {"type": "array", "items": {"type": "number"}}}},
                  open(os.path.join(self.folder, "ops/classify.input.json"), "w"))
        json.dump({"type": "object", "properties": {"label": {"type": "string"}}}, open(os.path.join(self.folder, "ops/classify.output.json"), "w"))
        open(os.path.join(self.folder, "README.md"), "w").write("# Toy Labels\n")
        open(os.path.join(self.folder, "log/001-add.md"), "w").write("+++\nverdict = \"kept\"\n+++\n")
        json.dump({"rows": [{"ops": {"classify": {"status": 200}}}]}, open(os.path.join(self.folder, "results/quick-checks/kit.json"), "w"))
        open(os.path.join(self.folder, "runners/swift/bin/runner"), "wb").write(b"\xcf\xfa\xed\xfe" + b"\0" * 60)
        shutil.copy(folder_check.KIT_PACKAGE, os.path.join(self.folder, "runners/swift/Package.swift"))
        self.model_swift = os.path.join(self.folder, "runners/swift/Sources/Runner/Model.swift")
        open(self.model_swift, "w").write(MODEL_SWIFT)

    def codes(self):
        report, _ = folder_check.run(self.folder, hub=self.hub)
        return [p["code"] for p in report.problems]

    def edit(self, old, new):
        text = open(self.toml).read()
        assert old in text, old
        open(self.toml, "w").write(text.replace(old, new))


class FolderChecks(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()

    def tearDown(self):
        shutil.rmtree(self.f.root)

    def test_valid_folder_passes(self):
        self.assertEqual(self.f.codes(), [])

    def test_pinned_torch_archive_needs_the_kit_cast_and_a_plain_state_dict(self):
        import pickle, zipfile
        from test_kit import bf16, write_torch_checkpoint
        snapshot = os.path.join(self.f.hub, "models--owner--toy", "snapshots", "a" * 40)
        os.remove(os.path.join(snapshot, "model.safetensors"))
        archive = os.path.join(snapshot, "model.pth")
        write_torch_checkpoint(archive, {"classifier.weight": ([3, 4], "BFloat16", bf16([0.5] * 12)),
                                         "classifier.bias": ([3], "BFloat16", bf16([0.0] * 3))})

        def pin():
            text = open(self.f.toml).read()
            text = re.sub(r'"model\.(safetensors|pth)" = "[0-9a-f]{64}"', f'"model.pth" = "{folder_check.sha256(archive)}"', text)
            text = re.sub(r'"model\.(safetensors|pth)" = [0-9]+', f'"model.pth" = {os.path.getsize(archive)}', text)
            open(self.f.toml, "w").write(text.replace('files = ["model.safetensors"', 'files = ["model.pth"'))
        pin()
        self.f.edit('dtype = "fp32"', 'dtype = "bf16"')
        self.assertEqual(self.f.codes(), ["S3-format"])  # nothing converts it
        open(self.f.toml, "a").write('\n[derive]\nfp16 = "cast:arsenal-kit-half-v1"\n')
        self.assertEqual(self.f.codes(), [])  # 15 BF16 values, read from the persistent ids
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as handle:
            handle.writestr("archive/data.pkl", pickle.dumps({"w": os.getcwd}, protocol=2))
        pin()
        self.assertIn("S3-pickle", self.f.codes())
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
        pin()
        self.assertIn("S3-header", self.f.codes())

    def add_pin(self, name, data, derive=True):
        """Pin one more file beside model.safetensors (the fixture's snapshot), optionally with the kit cast."""
        snapshot = os.path.join(self.f.hub, "models--owner--toy", "snapshots", "a" * 40)
        path = os.path.join(snapshot, name)
        open(path, "wb").write(data)
        text = open(self.f.toml).read()
        text = text.replace('files = [', f'files = ["{name}", ', 1)
        text = text.replace('[weights.sha256]\n', f'[weights.sha256]\n"{name}" = "{folder_check.sha256(path)}"\n')
        text = text.replace('[weights.bytes]\n', f'[weights.bytes]\n"{name}" = {len(data)}\n')
        if derive and "[derive]" not in text:
            text += '\n[derive]\nfp16 = "cast:arsenal-kit-half-v1"\n'
        open(self.f.toml, "w").write(text)
        return path

    def test_pinned_numpy_files_need_the_kit_cast_and_a_numeric_header(self):
        # Review item 6: .npy/.npz are checked as the converter reads them, and need the kit cast like .pth.
        from test_kit import write_npy
        good = write_npy(None, [2, 2], [1.0, 2.0, 3.0, 4.0])
        self.add_pin("idf.npy", good, derive=False)
        self.assertEqual(self.f.codes(), ["S3-format"])
        open(self.f.toml, "a").write('\n[derive]\nfp16 = "cast:arsenal-kit-half-v1"\n')
        self.assertEqual(self.f.codes(), [])
        objects = good.replace(b"'<f8'", b"'|O8'")
        self.add_pin("objects.npy", objects)
        self.assertEqual(self.f.codes(), ["S3-header"])  # an object array would need pickle

    def test_numpy_shape_must_match_its_data_and_npz_must_be_stored(self):
        import io, zipfile
        from test_kit import write_npy
        short = write_npy(None, [3], [1.0, 2.0, 3.0])[:-8]
        self.add_pin("short.npy", short)
        self.assertEqual(self.f.codes(), ["S3-header"])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("0.npy", write_npy(None, [1], [1.0]))
        self.f = Fixture()
        self.add_pin("planes.npz", buffer.getvalue())
        self.assertEqual(self.f.codes(), ["S3-header"])

    def torch_pin(self, state=None, raw=None):
        from test_kit import write_raw_archive, write_torch_checkpoint
        snapshot = os.path.join(self.f.hub, "models--owner--toy", "snapshots", "a" * 40)
        temporary = os.path.join(snapshot, "t.pth")
        if raw is not None:
            write_raw_archive(temporary, raw)
        else:
            write_torch_checkpoint(temporary, state)
        data = open(temporary, "rb").read()
        os.remove(temporary)
        return self.add_pin("extra.pth", data)

    def test_torch_tensors_must_be_contiguous_inside_their_storage(self):
        # Review item 7: _rebuild_tensor_v2's contiguity and bounds are checked as the converter checks them.
        from test_kit import bf16
        self.torch_pin({"w": ([2, 2], "BFloat16", bf16([1, 2, 3, 4]), [1, 2])})  # transposed strides
        self.assertIn("S3-header", self.f.codes())
        self.f = Fixture()
        self.torch_pin({"w": ([4, 4], "BFloat16", bf16([1, 2, 3, 4]))})  # 16 values in a 4-value storage
        self.assertIn("S3-header", self.f.codes())
        self.f = Fixture()
        self.torch_pin({"w": ([2], "Double", struct.pack("<2d", 1, 2))})  # float64 is refused
        self.assertIn("S3-header", self.f.codes())

    def test_malformed_and_hostile_pickles_are_reported_not_crashed(self):
        # Review item 7: the runner's opcode set, its depth and memo limits, and any malformed file is a report.
        self.torch_pin(raw=b"\x80\x02\xff.")  # an opcode the converter does not know
        self.assertEqual(self.f.codes(), ["S3-header"])
        self.f = Fixture()
        self.torch_pin(raw=b"\x80\x02)q\x00" + b"h\x00h\x00h\x00\x87q\x00" * 80 + b".")  # shared deep tuples
        self.assertEqual(self.f.codes(), ["S3-header"])
        self.f = Fixture()
        self.torch_pin(raw=b"\x80\x02}q\x00)K\x01s.")  # a tuple as a dictionary key
        self.assertEqual(self.f.codes(), ["S3-header"])
        self.f = Fixture()
        self.add_pin("broken.pth", b"not a zip archive at all")
        self.assertEqual(self.f.codes(), ["S3-header"])

    def test_data_pkl_over_the_limit_is_refused(self):
        from test_kit import bf16
        original = folder_check.PICKLE_LIMIT
        folder_check.PICKLE_LIMIT = 16  # the 64 MB limit, scaled down so the test needs no large file
        try:
            self.torch_pin({"w": ([2], "BFloat16", bf16([1, 2]))})
            self.assertEqual(self.f.codes(), ["S3-header"])
        finally:
            folder_check.PICKLE_LIMIT = original

    def test_placeholder_and_purpose(self):
        self.f.edit('name = "Toy Labels 1.0"', 'name = "TODO: full official name"')
        self.f.edit('purpose = "Label numbers"', 'purpose = "Labels four numbers into three classes, as a toy for tests"')
        self.assertEqual(sorted(self.f.codes()), ["S1-field", "S1-todo"])

    def test_malformed_toml(self):
        open(self.f.toml, "a").write("this is = = not toml [[\n")
        self.assertEqual(self.f.codes(), ["S1-toml"])

    def test_bad_facets_and_id(self):
        self.f.edit('modalities_in = ["text"]', 'modalities_in = ["pictures"]')
        self.f.edit('id = "toy-labels"', 'id = "other"')
        self.assertEqual(sorted(self.f.codes()), ["S1-facets", "S1-id"])

    def test_missing_weight_file(self):
        self.f.edit('files = ["model.safetensors", "config.json"]', 'files = ["model.safetensors", "config.json", "extra.safetensors"]')
        self.assertEqual(sorted(self.f.codes()), ["S3-missing", "S3-pin"])

    def test_params_and_dtype_against_header(self):
        self.f.edit('params = "15"', 'params = "7M"')
        self.f.edit('dtype = "fp32"', 'dtype = "fp16"')
        self.assertEqual(sorted(self.f.codes()), ["S3-dtype", "S3-params"])

    def test_wrong_op_schema(self):
        json.dump({"type": "object", "required": ["nothere"], "properties": {"values": {"type": "strng"}}},
                  open(os.path.join(self.f.folder, "ops/classify.input.json"), "w"))
        self.assertEqual(sorted(self.f.codes()), ["S2-required", "S2-type"])

    def test_schema_not_json_and_missing_output(self):
        open(os.path.join(self.f.folder, "ops/classify.input.json"), "w").write("{nope")
        os.remove(os.path.join(self.f.folder, "ops/classify.output.json"))
        self.assertEqual(sorted(self.f.codes()), ["S2-json", "S2-schema"])

    def test_file_input_needs_annotation(self):
        json.dump({"type": "object", "properties": {"image": {"type": "object", "properties": {"path": {"type": "string"}}}}},
                  open(os.path.join(self.f.folder, "ops/classify.input.json"), "w"))
        self.assertEqual(self.f.codes(), ["S2-file"])

    def test_no_runner_declared_or_built(self):
        os.remove(os.path.join(self.f.folder, "runners/swift/bin/runner"))
        self.assertEqual(self.f.codes(), ["S4-build"])
        text = open(self.f.toml).read()
        start = text.index("[[runner]]")
        open(self.f.toml, "w").write(text[:start] + text[text.index("[[op]]"):])
        self.assertEqual(self.f.codes(), ["S4-runner"])

    def test_script_runner_and_other_engine_refused(self):
        self.f.edit('language = "swift"', 'language = "python"')
        self.assertEqual(self.f.codes(), ["S4-mlx"])

    def test_python_and_processes_refused_in_sources(self):
        cases = {
            'let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")': 2,  # a process, and Python
            "let session = URLSession.shared": 1,
            "import CoreML": 1,
            'let h = dlopen("x", 0)': 1,
        }
        for line, count in cases.items():
            open(self.f.model_swift, "w").write(MODEL_SWIFT + line + "\n")
            self.assertEqual(self.f.codes(), ["S4-mlx"] * count, line)
        open(self.f.model_swift, "w").write(MODEL_SWIFT + "// a comment may mention python and Process( freely\n")
        self.assertEqual(self.f.codes(), [])

    def test_folder_may_not_carry_scripts_or_other_weights(self):
        open(os.path.join(self.f.folder, "serve.py"), "w").write("print(1)\n")
        open(os.path.join(self.f.folder, "model.onnx"), "wb").write(b"0")
        self.assertEqual(self.f.codes(), ["S4-mlx", "S4-mlx"])

    def test_kind_ops_package_and_import(self):
        open(self.f.model_swift, "w").write(MODEL_SWIFT.replace("import MLX\n", "").replace('"local.labels"', '"local.other"')
                                            .replace('["classify"]', '["classify", "extra"]'))
        open(os.path.join(self.f.folder, "runners/swift/Package.swift"), "a").write("// changed\n")
        self.assertEqual(sorted(self.f.codes()), ["S4-kind", "S4-mlx", "S4-ops", "S4-package"])

    def test_every_op_needs_a_measured_try(self):
        json.dump({"rows": [{"ops": {"other": {}}}]}, open(os.path.join(self.f.folder, "results/quick-checks/kit.json"), "w"))
        self.assertEqual(self.f.codes(), ["S7-tried"])

    def test_licence_open_local_or_refused(self):
        self.f.edit('spdx = "MIT"', 'spdx = "LicenseRef-other"')
        self.assertEqual(self.f.codes(), ["S6-licence"])  # not open and not marked as the owner's own
        self.f.edit("commercial = true", "commercial = false")
        self.assertEqual(self.f.codes(), ["S6-local-only"])  # a note: stays on this Mac


    def test_ops_may_live_in_the_embedded_kind(self):
        text = open(self.f.toml).read()
        open(self.f.toml, "w").write(text[:text.index("[[op]]")])
        kind_ops = os.path.join(self.f.folder, "kinds/local.labels/ops")
        os.makedirs(kind_ops)
        for side in ("input", "output"):
            os.rename(os.path.join(self.f.folder, f"ops/classify.{side}.json"), os.path.join(kind_ops, f"classify.{side}.json"))
        open(os.path.join(self.f.folder, "kinds/local.labels.toml"), "w").write(
            'id = "local.labels"\ntitle = "Labels"\n\n[[op]]\nname = "classify"\ninput = "ops/classify.input.json"\n'
            'output = "json"\noutput_schema = "ops/classify.output.json"\n')
        self.assertEqual(self.f.codes(), [])
        os.remove(os.path.join(kind_ops, "classify.output.json"))
        self.assertEqual(self.f.codes(), ["S2-schema"])


class TomlReader(unittest.TestCase):
    def test_matches_tomllib_on_every_shipped_folder(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib needs Python 3.11")
        names = sorted(os.listdir(os.path.join(ROOT, "models")))
        self.assertGreater(len(names), 20)
        for name in names:
            path = os.path.join(ROOT, "models", name, "model.toml")
            with open(path, "rb") as handle:
                expected = tomllib.load(handle)
            with mock.patch.dict(sys.modules, {"tomllib": None}):
                self.assertEqual(folder_check.read_toml(path), expected, name)


class KitInputs(unittest.TestCase):
    def test_repo_from_link_or_name(self):
        for text in ("owner/repo", "https://huggingface.co/owner/repo", "https://huggingface.co/owner/repo/tree/main",
                     "huggingface.co/owner/repo/blob/main/README.md", "https://hf.co/owner/repo/", " owner/repo "):
            self.assertEqual(kit.repo_id(text), "owner/repo", text)
        for text in ("https://huggingface.co/datasets/owner/data", "just-a-name"):
            with self.assertRaises(SystemExit):
                kit.repo_id(text)

    def test_scaffold_refuses_a_catalog_model_id(self):
        # A local folder named like a signed model is ignored by Arsenal, so the kit refuses the id up front (2 Nov).
        args = type("A", (), {"id": "flux3-image-klein", "kind": "images", "op": "generate", "input": None})()
        with tempfile.TemporaryDirectory() as work, mock.patch.object(kit, "signed_ids", return_value={"flux3-image-klein"}), \
                mock.patch.object(kit, "folder", return_value=os.path.join(work, "x")), mock.patch.object(kit, "SUPPORT", work):
            with self.assertRaises(SystemExit):
                kit.scaffold(args)

    def test_licence_acceptance_is_exact(self):
        args = type("A", (), {"accept_licence": None})()
        self.assertTrue(kit.accepted("apache-2.0", args))
        self.assertFalse(kit.accepted("other", args))
        args.accept_licence = "other"
        self.assertTrue(kit.accepted("other", args))
        args.accept_licence = "mit"
        with self.assertRaises(SystemExit):
            kit.accepted("other", args)

    def test_gated_repository_stops_with_instructions(self):
        error = urllib.error.HTTPError("https://huggingface.co/api/models/owner/gated", 401, "Unauthorized", {}, io.BytesIO())
        with mock.patch("urllib.request.urlopen", side_effect=error), mock.patch.object(kit, "hf_token", return_value=None), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, self.assertRaises(SystemExit):
            kit.hf_open("https://huggingface.co/api/models/owner/gated?blobs=true")
        self.assertIn("owner/gated is gated or private", out.getvalue())
        self.assertIn("none is set", out.getvalue())

    def test_token_is_sent_to_hugging_face(self):
        seen = {}

        def fake(request, timeout):
            seen["auth"] = request.get_header("Authorization")
            return io.BytesIO(b"{}")
        with mock.patch("urllib.request.urlopen", side_effect=fake), mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            kit.hf_open("https://huggingface.co/api/models/owner/repo")
        self.assertEqual(seen["auth"], "Bearer hf_test")

    def test_second_op_keeps_the_first_ops_record(self):
        work = tempfile.mkdtemp(prefix="arsenal-record-")
        try:
            target = os.path.join(work, "models", "toy")
            os.makedirs(os.path.join(target, "runners/swift/bin"))
            os.makedirs(os.path.join(target, "log"))
            open(os.path.join(target, "runners/swift/bin/runner"), "wb").write(b"runner")
            open(os.path.join(target, "model.toml"), "w").write('kind = "local.toy"\n[weights]\nrepo = "o/r"\nrevision = "' + "b" * 40 + '"\n')
            with mock.patch.object(kit, "WORK", work), mock.patch.object(kit, "app", return_value=work):
                kit.record("toy", "first", 0.1, [0.2, 0.1], 100)
                kit.record("toy", "second", 0.1, [0.3, 0.2], 80)
            records = os.listdir(os.path.join(target, "results/quick-checks"))
            self.assertEqual(len(records), 1)
            row = json.load(open(os.path.join(target, "results/quick-checks", records[0])))["rows"][0]
            self.assertEqual(sorted(row["ops"]), ["first", "second"])
        finally:
            shutil.rmtree(work)


if __name__ == "__main__":
    unittest.main()
