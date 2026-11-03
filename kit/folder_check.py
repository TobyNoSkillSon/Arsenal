# SPDX-License-Identifier: MIT
"""Static checks of a model folder: everything Arsenal can tell about a new model without running it.

    python3 kit/new-model.py check ID            the kit's working folder for ID
    python3 kit/new-model.py check /path/folder  any folder, e.g. one already in local-models/

Each problem is one line: a code, the file and field, what is wrong, and how to fix it. Nothing here runs the model;
the optional weights check (`--load`) starts the runner to load the pinned weights into its modules and runs no forward pass.
Standard library only (Python 3.9+, as on a fresh Mac).
"""
import ast, hashlib, json, os, pickletools, re, struct, zipfile

OPEN_SPDX = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "CC0-1.0", "CC-BY-4.0", "MPL-2.0", "Unlicense", "Zlib"}
MODALITIES = {"text", "audio", "image", "video"}
JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
# Python never serves a model, and a runner never shells out, loads code at run time or reaches the network.
FORBIDDEN = [
    (r"\bProcess\s*\(|\bNSTask\b|posix_spawn|\bsystem\s*\(|\bpopen\s*\(|\bexecv?p?\s*\(", "starts another program"),
    (r"\bpython[0-9.]*\b|\.py\b|PythonKit|import\s+Python", "uses Python"),
    (r"\bdlopen\s*\(|\bdlsym\s*\(|Bundle\s*\(\s*path", "loads code at run time"),
    (r"\bURLSession\b|import\s+Network\b|\bNWConnection\b|CFNetwork|https?://", "reaches the network"),
    (r"import\s+(CoreML|Metal(?!Kit)|Accelerate|onnx|ONNX|TensorFlow|Torch)\b", "computes outside MLX"),
]
NOT_IN_FOLDER = (".py", ".pyc", ".sh", ".ipynb", ".onnx", ".pt", ".pth", ".bin", ".gguf", ".mlmodel", ".mlpackage", ".so", ".dylib")
KIT_PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner", "Package.swift")


# ---- a reader for the TOML that model folders use (tomllib when present; Python 3.9 has none) ----------------------------

def _value(text):
    text = text.strip()
    if text in ("true", "false"):
        return text == "true"
    if text.startswith("{"):
        out = {}
        for part in _split(text[1:-1]):
            key, _, value = part.partition("=")
            out[key.strip().strip('"')] = _value(value)
        return out
    if text.startswith("["):
        return [_value(part) for part in _split(text[1:-1]) if part.strip()]
    if text.startswith('"'):
        return ast.literal_eval(text)
    if text.startswith("'"):
        return text[1:-1]
    try:
        return int(text.replace("_", ""))
    except ValueError:
        return float(text)


def _split(text):
    """Top-level commas only: not inside strings, arrays or inline tables."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\" and quote == '"':
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i]); start = i + 1
        i += 1
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


def _strip_comment(line):
    quote = None
    for i, c in enumerate(line):
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i]
    return line


def read_toml(path):
    try:
        import tomllib
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except ImportError:
        pass
    root, table, pending = {}, None, ""
    table = root
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for number, raw in enumerate(lines, 1):
        line = _strip_comment(raw).strip()
        if pending:
            line, pending = pending + " " + line, ""
        if not line:
            continue
        if line.count("[") > line.count("]") or line.count("{") > line.count("}"):
            pending = line
            continue
        if line.startswith("[["):
            names = [n.strip().strip('"') for n in line[2:-2].split(".")]
            parent = root
            for name in names[:-1]:
                parent = parent.setdefault(name, {})
                parent = parent[-1] if isinstance(parent, list) else parent
            parent.setdefault(names[-1], []).append({})
            table = parent[names[-1]][-1]
        elif line.startswith("["):
            table = root
            for name in [n.strip().strip('"') for n in re.findall(r'"[^"]*"|[^.]+', line[1:-1])]:
                table = table.setdefault(name, {})
                table = table[-1] if isinstance(table, list) else table
        else:
            key, sep, value = line.partition("=")
            if not sep:
                raise ValueError(f"line {number}: expected key = value")
            table[key.strip().strip('"')] = _value(value)
    if pending:
        raise ValueError(f"an unclosed [ or {{ near: {pending[:60]}")
    return root


# ---- the checks ------------------------------------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.problems = []

    def add(self, code, where, what, fix):
        self.problems.append({"code": code, "where": where, "problem": what, "fix": fix})

    def lines(self):
        return [f"{p['code']} {p['where']}: {p['problem']} Fix: {p['fix']}" for p in self.problems]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size > 100_000_000:
            raise ValueError(f"header claims {size} bytes")
        return json.loads(handle.read(size))


def params_from_label(label):
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMBT]?)\s*", str(label or ""), re.I)
    if not match:
        return None
    return float(match.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[match.group(2).upper()]


def check_manifest(folder, m, report):
    fid = os.path.basename(os.path.normpath(folder))
    if m.get("id") != fid:
        report.add("S1-id", "model.toml id", f"is {m.get('id')!r} but the folder is named {fid!r}.", f'set id = "{fid}" or rename the folder.')
    if m.get("schema") != 3:
        report.add("S1-schema", "model.toml schema", f"is {m.get('schema')!r}; a local folder is schema 3.", "set schema = 3.")
    for key in ("name", "summary", "description", "purpose", "params", "publisher"):
        if not str(m.get(key) or "").strip():
            report.add("S1-field", f"model.toml {key}", "is empty.", "fill it in from the model card.")
    if len(str(m.get("purpose") or "")) > 40:
        report.add("S1-field", "model.toml purpose", f"is {len(m['purpose'])} characters; at most 40.", "a few words, e.g. Find body keypoints.")
    for key in ("languages", "good_for", "limits", "tasks"):
        if not m.get(key):
            report.add("S1-field", f"model.toml {key}", "is empty.", "list at least one entry from the model card.")
    facets = m.get("facets") or {}
    for key in ("modalities_in", "modalities_out"):
        values = facets.get(key) or []
        bad = [v for v in values if v not in MODALITIES]
        if not values or bad or len(set(values)) != len(values):
            report.add("S1-facets", f"model.toml [facets] {key}", f"is {values}; each must be one of text, audio, image, video, once.", "list what the model takes in and gives out.")
    for root, _, files in os.walk(folder):
        if "/runners/swift/bin" in root or "/runners/swift/.build" in root:
            continue
        for name in files:
            path = os.path.join(root, name)
            if name.endswith((".toml", ".md", ".json", ".swift")):
                text = open(path, errors="ignore").read()
                for match in re.finditer(r"\bTODO\b.{0,60}", text):
                    report.add("S1-todo", os.path.relpath(path, folder), f"still has a placeholder: {match.group(0).strip()!r}.", "replace it with the real value.")
                    break
    if not os.path.isdir(os.path.join(folder, "log")) or not any(f.endswith(".md") for f in os.listdir(os.path.join(folder, "log"))):
        report.add("S1-log", "log/", "has no entry; Arsenal needs one.", "run kit/new-model.py try, which writes log/001-add.md.")


def check_licence(m, report):
    licence = m.get("licence") or {}
    spdx = str(licence.get("spdx") or "")
    if not spdx:
        report.add("S6-licence", "model.toml [licence] spdx", "is missing.", "state the licence from the model card.")
    elif spdx not in OPEN_SPDX:
        if not spdx.startswith("LicenseRef-") or licence.get("commercial") is not False:
            report.add("S6-licence", "model.toml [licence]", f"{spdx} is not an open licence, and the folder does not mark it as accepted locally.",
                       "only with the user's acceptance: rerun kit/new-model.py scaffold with --accept-licence (spdx LicenseRef-…, commercial = false).")
        else:
            report.add("S6-local-only", "model.toml [licence]", f"{spdx} is not open: this model stays on this Mac. Arsenal never lists, ships or shares it.",
                       "nothing to fix; never copy this folder into Arsenal's catalogue.")


def resolved_ops(folder, m):
    """The model's operations: its own [[op]], else those of its embedded local kind (kind_file), with paths from the folder."""
    if m.get("op"):
        return m["op"], ""
    kind_file = m.get("kind_file")
    path = os.path.join(folder, kind_file) if kind_file else None
    if path and os.path.exists(path):
        try:
            return read_toml(path).get("op") or [], os.path.join(os.path.dirname(kind_file), str(m.get("kind")))
        except Exception:
            return [], ""
    return [], ""


def check_ops(folder, m, kinds_dir, report):
    ops, base = resolved_ops(folder, m)
    if not ops:
        report.add("S2-op", "model.toml [[op]]", "no operation is declared, here or in the kind_file.", "declare at least one [[op]] with name, input and output.")
    kind = str(m.get("kind") or "")
    local_kind = kind.startswith("local.")
    for op in ops:
        name = op.get("name", "?")
        for side, key in (("input", "input"), ("output", "output_schema")):
            ref = op.get(key)
            if ref is None and side == "output":
                ref = op.get("output") if str(op.get("output", "")).endswith(".json") else None
            if ref is None:
                if local_kind or side == "input":
                    path = os.path.join(folder, f"ops/{name}.{side}.json")
                    if not os.path.exists(path) and not (kinds_dir and os.path.exists(os.path.join(kinds_dir, kind, f"ops/{name}.{side}.json"))):
                        report.add("S2-schema", f"[[op]] {name}", f"has no {side} schema.", f"write ops/{name}.{side}.json.")
                continue
            path = os.path.join(folder, base, ref)
            if not os.path.exists(path) and kinds_dir:
                path = os.path.join(kinds_dir, kind, ref)
            if not os.path.exists(path):
                report.add("S2-schema", f"[[op]] {name} {key}", f"names {ref}, which does not exist.", "write the file or fix the path.")
                continue
            check_schema(os.path.relpath(path, folder) if path.startswith(folder) else ref, path, side, report)


def check_schema(where, path, side, report):
    try:
        schema = json.load(open(path))
    except ValueError as error:
        report.add("S2-json", where, f"is not JSON ({error}).", "fix the syntax.")
        return
    if os.path.getsize(path) > 1 << 20:
        report.add("S2-size", where, "is over 1 MiB.", "shorten it.")

    def walk(node, at):
        if not isinstance(node, dict):
            return
        kinds = node.get("type")
        for t in (kinds if isinstance(kinds, list) else [kinds] if kinds else []):
            if t not in JSON_TYPES:
                report.add("S2-type", f"{where} {at}", f'has type "{t}".', f"use one of {', '.join(sorted(JSON_TYPES))}.")
        props = node.get("properties") or {}
        for key in node.get("required") or []:
            if props and key not in props:
                report.add("S2-required", f"{where} {at}", f'requires "{key}", which is not in its properties.', "add the property or drop it from required.")
        if side == "input" and isinstance(props.get("path"), dict) and set(props) == {"path"} and "x-arsenal-file" not in node:
            report.add("S2-file", f"{where} {at}", "is a file object without x-arsenal-file, so Arsenal will not accept a picked file.",
                       'add "x-arsenal-file": {"accept": ["image/png", …]}.')
        for key, child in props.items():
            walk(child, f"{at}.{key}")
        if isinstance(node.get("items"), dict):
            walk(node["items"], at + "[]")
        for key in ("$defs", "definitions"):
            for name, child in (node.get(key) or {}).items():
                walk(child, f"{at}/{key}/{name}")
        for key in ("oneOf", "anyOf", "allOf"):
            for index, child in enumerate(node.get(key) or []):
                walk(child, f"{at}/{key}/{index}")

    if not isinstance(schema, dict) or schema.get("type") != "object":
        report.add("S2-type", where, "the top level must be an object schema.", 'start with {"type": "object", "properties": {…}}.')
    walk(schema, "$")


# Pinned PyTorch archives (.pth, .pt) and NumPy files (.npy, .npz) are not loaded: the kit runner's `convert` turns them
# into safetensors at Get (cast:arsenal-kit-half-v1), reading the pickle without running it. This check reads them the
# same way, with the runner's opcode set and limits (kit/runner/Sources/Runner/Checkpoints.swift), so a file passes
# here exactly when the converter would take it.
TORCH_SUFFIXES = (".pth", ".pt")
NUMPY_SUFFIXES = (".npy", ".npz")
CONVERTED_SUFFIXES = TORCH_SUFFIXES + NUMPY_SUFFIXES
KIT_CAST = "cast:arsenal-kit-half-v1"
TORCH_GLOBALS = {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_parameter"}
TORCH_STORAGES = {"FloatStorage": ("F32", 4), "HalfStorage": ("F16", 2), "BFloat16Storage": ("BF16", 2), "DoubleStorage": ("F64", 8),
                  "LongStorage": ("I64", 8), "IntStorage": ("I32", 4), "ShortStorage": ("I16", 2), "CharStorage": ("I8", 1),
                  "ByteStorage": ("U8", 1), "BoolStorage": ("BOOL", 1)}
NUMPY_TYPES = {"<f8": ("F64", 8), "<f4": ("F32", 4), "<f2": ("F16", 2), "<i8": ("I64", 8), "<i4": ("I32", 4), "<i2": ("I16", 2),
               "|i1": ("I8", 1), "<u8": ("U64", 8), "<u4": ("U32", 4), "<u2": ("U16", 2), "|u1": ("U8", 1), "|b1": ("BOOL", 1)}
PICKLE_LIMIT = 64 << 20
MAX_DEPTH, MAX_VALUES, MAX_MEMO, MAX_INT = 64, 8_000_000, 1_000_000, (1 << 63) - 1


class Refused(ValueError):
    """A file the converter would refuse, with the reason it would give."""


def _stored_zip(path):
    archive = zipfile.ZipFile(path)
    if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
        raise Refused("its entries are compressed; the converter reads stored entries only")
    return archive


def parse_torch_pickle(data, globals_seen=None):
    """The runner's TorchPickle.parse in Python: the same opcodes and bounds. Returns the root value. Containers are
    lists [kind, items, depth, frozen]; tuples ("tuple", items, depth); scalars as Python values."""
    stack, marks, memo, pushed, i = [], [], {}, 0, 0

    def push(value):
        nonlocal pushed
        pushed += 1
        if pushed > MAX_VALUES:
            raise Refused("too many values")
        stack.append(value)

    def depth(value):
        if isinstance(value, tuple) and value[:1] == ("tuple",):
            return value[2]
        if isinstance(value, list):
            return value[2]
        return 0

    def nest(value):
        if isinstance(value, list):
            value[3] = True

    def make_tuple(items):
        d = 1 + max([depth(v) for v in items], default=0)
        if d > MAX_DEPTH:
            raise Refused("nesting too deep")
        for v in items:
            nest(v)
        return ("tuple", items, d)

    def take(n):
        nonlocal i
        if n < 0 or n > len(data) - i:
            raise Refused("truncated pickle")
        chunk = data[i:i + n]
        i += n
        return chunk

    def uint(n):
        return int.from_bytes(take(n), "little")

    def pop():
        if len(stack) <= (marks[-1] if marks else 0):
            raise Refused("stack underflow")
        return stack.pop()

    def pop_mark():
        if not marks or marks[-1] > len(stack):
            raise Refused("missing mark")
        index = marks.pop()
        items = stack[index:]
        del stack[index:]
        return items

    def target():
        if not stack or len(stack) <= (marks[-1] if marks else 0):
            raise Refused("no container to fill")
        return stack[-1]

    def fill(box, kind, values):
        if not isinstance(box, list) or box[0] != kind:
            raise Refused(f"{kind} expected")
        for value in values:
            if box[3]:
                raise Refused(f"a {kind} changes after it was nested")
            d = 1 + depth(value)
            if d > MAX_DEPTH:
                raise Refused("nesting too deep")
            nest(value)
            if box[3]:
                raise Refused(f"a {kind} changes after it was nested")
            box[2] = max(box[2], d)
            box[1].append(value)

    def set_items(items, box):
        if len(items) % 2:
            raise Refused("odd SETITEMS")
        for key in items[0::2]:
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise Refused("a dictionary key that is not a string or an integer")
        fill(box, "dict", [(items[k], items[k + 1]) for k in range(0, len(items), 2)])

    def remember(key):
        if not stack or len(stack) <= (marks[-1] if marks else 0):
            raise Refused("memo of nothing")
        if key not in memo and len(memo) >= MAX_MEMO:
            raise Refused("too many memo entries")
        memo[key] = stack[-1]

    def reduce(callable_, args):
        if not (isinstance(callable_, tuple) and callable_[:1] == ("global",) and isinstance(args, tuple) and args[:1] == ("tuple",)):
            return ("opaque", "call")
        name, items = callable_[1], args[1]
        if name == "collections.OrderedDict":
            return ["dict", [], 1, False]
        if name == "torch._utils._rebuild_parameter":
            return items[0] if items else None
        if name == "torch._utils._rebuild_tensor_v2":
            if len(items) < 4 or not (isinstance(items[0], tuple) and items[0][:1] == ("storage",)) or not isinstance(items[1], int):
                raise Refused("malformed tensor")
            ints = []
            for part in items[2:4]:
                if not (isinstance(part, tuple) and part[:1] == ("tuple",) and all(isinstance(v, int) and not isinstance(v, bool) for v in part[1])):
                    raise Refused("malformed tensor shape or stride")
                ints.append(list(part[1]))
            if len(ints[0]) != len(ints[1]):
                raise Refused("tensor rank")
            return ("tensor", items[0][1], items[0][2], items[1], ints[0], ints[1])
        return ("opaque", name)

    def text(raw):
        return raw.decode("utf-8", errors="replace")

    while True:
        if i >= len(data):
            raise Refused("missing STOP")
        op = data[i]
        i += 1
        if op == 0x80: take(1)
        elif op == 0x95: take(8)
        elif op == 0x2E:
            if marks or len(stack) != 1:
                raise Refused("STOP with values left over")
            return pop()
        elif op == 0x28:
            if len(marks) >= MAX_DEPTH * 4:
                raise Refused("too many open marks")
            marks.append(len(stack))
        elif op == 0x7D: push(["dict", [], 1, False])
        elif op == 0x5D: push(["list", [], 1, False])
        elif op == 0x29: push(("tuple", [], 1))
        elif op == 0x4E: push(None)
        elif op == 0x88: push(True)
        elif op == 0x89: push(False)
        elif op == 0x4B: push(uint(1))
        elif op == 0x4D: push(uint(2))
        elif op == 0x4A: push(int.from_bytes(take(4), "little", signed=True))
        elif op == 0x8A:
            n = uint(1)
            if n > 8:
                raise Refused("long too wide")
            push(int.from_bytes(take(n), "little", signed=True))
        elif op == 0x47: push(struct.unpack(">d", take(8))[0])
        elif op == 0x58: push(text(take(uint(4))))
        elif op == 0x8C: push(text(take(uint(1))))
        elif op == 0x43: push(bytes(take(uint(1))))
        elif op == 0x42: push(bytes(take(uint(4))))
        elif op == 0x55: push(text(take(uint(1))))
        elif op == 0x54: push(text(take(uint(4))))
        elif op == 0x63:
            end = data.find(b"\n", i)
            end2 = data.find(b"\n", end + 1) if end >= 0 else -1
            if end < 0 or end2 < 0:
                raise Refused("unterminated GLOBAL")
            module, name = text(data[i:end]), text(data[end + 1:end2])
            i = end2 + 1
            push(("global", module + "." + name))
            if globals_seen is not None:
                globals_seen.add(module + "." + name)
        elif op == 0x93:
            name, module = pop(), pop()
            if not (isinstance(name, str) and isinstance(module, str)):
                raise Refused("STACK_GLOBAL")
            push(("global", module + "." + name))
            if globals_seen is not None:
                globals_seen.add(module + "." + name)
        elif op == 0x71: remember(uint(1))
        elif op == 0x72: remember(uint(4))
        elif op == 0x94: remember(len(memo))
        elif op in (0x68, 0x6A):
            key = uint(1 if op == 0x68 else 4)
            if key not in memo:
                raise Refused("memo")
            push(memo[key])
        elif op == 0x74: push(make_tuple(pop_mark()))
        elif op == 0x85: push(make_tuple([pop()]))
        elif op == 0x86:
            b, a = pop(), pop()
            push(make_tuple([a, b]))
        elif op == 0x87:
            c, b, a = pop(), pop(), pop()
            push(make_tuple([a, b, c]))
        elif op == 0x52:
            args, callable_ = pop(), pop()
            push(reduce(callable_, args))
        elif op == 0x51:
            pid = pop()
            if not (isinstance(pid, tuple) and pid[:1] == ("tuple",) and len(pid[1]) >= 3 and pid[1][0] == "storage"
                    and isinstance(pid[1][1], tuple) and pid[1][1][:1] == ("global",) and isinstance(pid[1][2], str)):
                raise Refused("unsupported persistent id")
            storage_type = pid[1][1][1]
            if not storage_type.startswith("torch.") or storage_type[6:] not in TORCH_STORAGES:
                raise Refused(f"unsupported storage {storage_type}")
            push(("storage", pid[1][2], TORCH_STORAGES[storage_type[6:]]))
        elif op == 0x62: pop()
        elif op == 0x73:
            value, key = pop(), pop()
            set_items([key, value], target())
        elif op == 0x75:
            items = pop_mark()
            set_items(items, target())
        elif op == 0x61:
            value = pop()
            fill(target(), "list", [value])
        elif op == 0x65:
            items = pop_mark()
            fill(target(), "list", items)
        else:
            raise Refused(f"unsupported opcode 0x{op:02x}")


def torch_checkpoint_summary(path):
    """(values per dtype, unsupported globals) of a torch.save archive, read as the converter reads it: the pickle is
    walked without being run, every tensor must be contiguous and lie inside its storage, and float64 is refused.
    Raises Refused (a ValueError) for a file the converter would refuse."""
    with _stored_zip(path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        pickles = sorted(name for name in infos if name == "data.pkl" or name.endswith("/data.pkl"))
        if not pickles:
            raise Refused("it has no data.pkl, so it is not a torch.save archive")
        if infos[pickles[0]].file_size > PICKLE_LIMIT:
            raise Refused(f"its data.pkl is over {PICKLE_LIMIT >> 20} MB")
        data = archive.read(pickles[0])
        prefix = pickles[0][:-len("data.pkl")]
        names = set()
        root = parse_torch_pickle(data, names)
        tensors = []

        def walk(value, depth_):
            if depth_ > 16:
                raise Refused("checkpoint dictionaries are nested too deeply")
            if not (isinstance(value, list) and value[0] == "dict"):
                return
            for key, item in value[1]:
                if isinstance(item, tuple) and item[:1] == ("tensor",):
                    tensors.append(item)
                elif isinstance(item, list) and item[0] == "dict":
                    walk(item, depth_ + 1)

        walk(root, 0)
        allowed = TORCH_GLOBALS | {"torch." + t for t in TORCH_STORAGES}
        unsupported = sorted(g for g in names if g not in allowed)
        if not tensors:
            if unsupported:
                return {}, unsupported
            raise Refused("it holds no tensors")
        values = {}
        for _, key, (dtype, width), offset, shape, stride in tensors:
            if dtype == "F64":
                raise Refused(f"storage {key} holds float64 values, which the converter does not cast")
            entry = infos.get(prefix + "data/" + key)
            if entry is None:
                raise Refused(f"storage {key} is missing")
            count = 1
            for dim in shape:
                if dim < 0:
                    raise Refused(f"invalid tensor shape {shape}")
                count *= dim
            contiguous = 1
            for dim, step in reversed(list(zip(shape, stride))):
                if dim > 1:
                    if step != contiguous:
                        raise Refused(f"storage {key} holds a non-contiguous tensor")
                    contiguous *= dim
            if count > MAX_INT or contiguous > MAX_INT or offset < 0 or offset * width > entry.file_size or count * width > entry.file_size - offset * width:
                raise Refused(f"a tensor exceeds its storage {key}")
            values[dtype] = values.get(dtype, 0) + count
    return values, unsupported


def parse_npy(data, name):
    """The runner's NumpyArrays.npy header checks: version, little-endian numeric dtype, C order, a shape whose byte
    count matches the data. Returns (safetensors dtype, values)."""
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        raise Refused(f"{name} is not a .npy array")
    if data[6] == 1:
        length, start = int.from_bytes(data[8:10], "little"), 10
    elif data[6] in (2, 3) and len(data) >= 12:
        length, start = int.from_bytes(data[8:12], "little"), 12
    else:
        raise Refused(f"{name}: .npy version {data[6]} is not supported")
    if length > 1 << 16 or start + length > len(data):
        raise Refused(f"{name}: invalid header")
    header = data[start:start + length].decode("latin-1")
    descr = re.search(r"'descr':\s*'([^']*)'", header)
    order = re.search(r"'fortran_order':\s*(True|False)", header)
    shape = re.search(r"'shape':\s*\(([^)]*)\)", header)
    if not descr or descr.group(1) not in NUMPY_TYPES:
        raise Refused(f"{name} holds dtype {descr.group(1) if descr else '?'}; only little-endian numeric arrays convert")
    if not order or order.group(1) != "False":
        raise Refused(f"{name} is stored in Fortran order; only C order converts")
    if not shape:
        raise Refused(f"{name}: no shape in the header")
    dims = [d.strip() for d in shape.group(1).split(",") if d.strip()]
    if not all(d.isdigit() for d in dims):
        raise Refused(f"{name}: bad shape")
    count = 1
    for d in dims:
        count *= int(d)
    dtype, width = NUMPY_TYPES[descr.group(1)]
    if count * width > MAX_INT or len(data) - start - length != count * width:
        raise Refused(f"{name}: its data does not match its header's shape")
    return dtype, count


def numpy_summary(path):
    """(values per dtype) of a .npy or .npz file, as the converter reads it."""
    if path.endswith(".npy"):
        with open(path, "rb") as handle:
            dtype, count = parse_npy(handle.read(), os.path.basename(path))
        return {dtype: count}
    values = {}
    with _stored_zip(path) as archive:
        members = archive.namelist()
        if not members:
            raise Refused("it holds no arrays")
        for member in members:
            if not member.endswith(".npy"):
                raise Refused(f"it holds {member}, not a .npy array")
            dtype, count = parse_npy(archive.read(member), member)
            values[dtype] = values.get(dtype, 0) + count
    return values


def check_weights(m, hub, report):
    w = m.get("weights") or {}
    repo, revision, files = w.get("repo"), w.get("revision"), w.get("files") or []
    if not repo or not re.fullmatch(r"[0-9a-f]{40}", str(revision or "")):
        report.add("S3-pin", "model.toml [weights]", "needs repo and a 40-hex revision.", "rerun kit/new-model.py scaffold, which pins them.")
        return None
    hashes, sizes = w.get("sha256") or {}, w.get("bytes") or {}
    snapshot = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", revision)
    total, dtypes, cached = 0, {}, True
    for name in files:
        if name not in hashes or name not in sizes:
            report.add("S3-pin", f"model.toml [weights] {name}", "has no sha256 or bytes pin.", "rerun scaffold, or pin it from the Hugging Face file page.")
        path = os.path.join(snapshot, name)
        if not os.path.exists(path):
            cached = False
            report.add("S3-missing", f"model.toml [weights] files {name}",
                       f"is not in {repo} at {revision[:12]} as cached on this Mac, so Get would fail.",
                       "check the file list on Hugging Face; remove names the repo does not have, or run scaffold to download it.")
            continue
        if name in sizes and os.path.getsize(path) != sizes[name]:
            report.add("S3-size", f"[weights.bytes] {name}", f"pins {sizes[name]} bytes; the file has {os.path.getsize(path)}.", "rerun scaffold to repin.")
        elif name in hashes and len(str(hashes[name])) == 64 and sha256(path) != hashes[name]:
            report.add("S3-hash", f"[weights.sha256] {name}", "does not match the file.", "rerun scaffold to repin.")
        if name.endswith(".safetensors"):
            try:
                header = safetensors_header(path)
            except Exception as error:
                report.add("S3-header", name, f"has an unreadable safetensors header ({error}).", "pin a different file.")
                continue
            for key, tensor in header.items():
                if key == "__metadata__":
                    continue
                count = 1
                for d in tensor["shape"]:
                    count *= d
                total += count
                dtypes[tensor["dtype"]] = dtypes.get(tensor["dtype"], 0) + count
        elif name.endswith(CONVERTED_SUFFIXES):
            if (m.get("derive") or {}).get("fp16") != KIT_CAST:
                kind = "a PyTorch archive" if name.endswith(TORCH_SUFFIXES) else "a NumPy file"
                report.add("S3-format", f"model.toml [weights] files {name}", f"is {kind}, which only the kit's Get-time conversion reads.",
                           f'declare [derive] fp16 = "{KIT_CAST}", or pin safetensors.')
                continue
            try:
                if name.endswith(TORCH_SUFFIXES):
                    values, unsupported = torch_checkpoint_summary(path)
                else:
                    values, unsupported = numpy_summary(path), []
            except Exception as error:  # a malformed file is reported, never a crash of the check
                report.add("S3-header", name, f"cannot be converted: {error}.", "pin safetensors, a plain torch.save state dict or a numeric NumPy file.")
                continue
            if unsupported:
                report.add("S3-pickle", name, f"needs code to unpickle ({', '.join(unsupported[:4])}); the converter runs none.",
                           "pin a plain torch.save state dict or a safetensors release.")
            if name.endswith(TORCH_SUFFIXES):  # NumPy files are data, not weights: they do not count as parameters
                for dtype, count in values.items():
                    total += count
                    dtypes[dtype] = dtypes.get(dtype, 0) + count
    if not any(name.endswith((".safetensors",) + TORCH_SUFFIXES) for name in files):
        report.add("S3-format", "model.toml [weights] files", "has no .safetensors file. Arsenal loads safetensors only.",
                   "find a safetensors release of the same weights.")
    if total and dtypes:
        main = max(dtypes, key=dtypes.get)
        expected = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}.get(main)
        if expected and w.get("dtype") != expected:
            report.add("S3-dtype", "model.toml [weights] dtype", f"is {w.get('dtype')!r} but most tensors are {main}.", f'set dtype = "{expected}".')
        declared = params_from_label(m.get("params"))
        if declared and abs(declared - total) / total > 0.1:
            report.add("S3-params", "model.toml params", f"says {m.get('params')} but the weights hold {total:,} values.",
                       "use the plan's count, or the card's if it counts a shared or excluded part differently (say so in description).")
    return {"repo": repo, "revision": revision, "snapshot": snapshot, "values": total, "dtypes": dtypes, "cached": cached}


def check_runner(folder, m, report):
    runners = m.get("runner") or []
    if not runners:
        report.add("S4-runner", "model.toml [[runner]]", "no runner is declared, so Arsenal cannot run this model.", "scaffold with the kit, which declares the kit runner.")
        return
    for runner in runners:
        entry = runner.get("entry")
        if runner.get("language") != "swift" or not entry:
            report.add("S4-mlx", f"[[runner]] {runner.get('name')}", "is not a folder-local Swift runner. Added models run on MLX in Swift only.",
                       'use the kit runner: language = "swift", entry = "runners/swift/bin/runner".')
            continue
        path = os.path.join(folder, entry)
        if not os.path.exists(path):
            report.add("S4-build", entry, "does not exist.", "run kit/new-model.py try (or build), which compiles it.")
        else:
            with open(path, "rb") as handle:
                magic = handle.read(4)
            if magic not in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
                report.add("S4-binary", entry, "is not a Mach-O executable (a script cannot be a runner).", "build it with the kit.")
            if not os.path.isdir(os.path.join(os.path.dirname(path), "mlx-swift_Cmlx.bundle")):
                report.add("S4-shaders", os.path.dirname(entry), "has no mlx-swift_Cmlx.bundle beside the runner, so MLX has no GPU shaders.", "rebuild with the kit.")
    package = os.path.join(folder, "runners/swift/Package.swift")
    if os.path.exists(package) and os.path.exists(KIT_PACKAGE) and open(package).read() != open(KIT_PACKAGE).read():
        report.add("S4-package", "runners/swift/Package.swift", "differs from the kit's, so the MLX pin and dependencies are not the ones Arsenal checked.",
                   "restore the kit's Package.swift; put code only in Sources/Runner/.")
    sources = os.path.join(folder, "runners/swift/Sources")
    swift = []
    for root, _, names in os.walk(sources):
        swift += [os.path.join(root, n) for n in names if n.endswith(".swift")]
    if not swift:
        report.add("S4-source", "runners/swift/Sources", "has no Swift source; Arsenal shows every runner's code to the user for approval.", "keep the kit's sources.")
    kit_files = {"Kit.swift", "Wire.swift", "main.swift"}
    for path in swift:
        text = open(path, errors="ignore").read()
        code = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
        code_no_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
        if os.path.basename(path) in kit_files:
            continue
        for pattern, what in FORBIDDEN:
            hit = re.search(pattern, code if "http" in pattern or "python" in pattern else code_no_strings)
            if hit:
                line = code[:hit.start()].count("\n") + 1
                report.add("S4-mlx", f"{os.path.relpath(path, folder)}:{line}", f"{what} ({hit.group(0)!r}). A runner computes only with MLX in this process.",
                           "port that part to MLX Swift; do the work in Model.swift.")
    model = os.path.join(sources, "Runner/Model.swift")
    if os.path.exists(model):
        text = open(model, errors="ignore").read()
        if not re.search(r"^\s*import\s+MLX\b", text, re.M):
            report.add("S4-mlx", "runners/swift/Sources/Runner/Model.swift", "does not import MLX.", "compute with MLX (import MLX, MLXNN).")
        kind = re.search(r'let\s+modelKind\s*=\s*"([^"]+)"', text)
        if kind and kind.group(1) != m.get("kind"):
            report.add("S4-kind", "Model.swift modelKind", f"is {kind.group(1)!r}; model.toml kind is {m.get('kind')!r}.", f'set let modelKind = "{m.get("kind")}".')
        ops = re.search(r"let\s+modelOps\s*=\s*\[([^\]]*)\]", text)
        declared = sorted(op.get("name") for op in resolved_ops(folder, m)[0])
        if ops and sorted(re.findall(r'"([^"]+)"', ops.group(1))) != declared:
            report.add("S4-ops", "Model.swift modelOps", f"is [{ops.group(1).strip()}]; model.toml declares {declared}.", "list exactly the [[op]] names.")
    for root, _, names in os.walk(folder):
        if "/runners/swift/bin" in root:
            continue
        for name in names:
            if name.lower().endswith(NOT_IN_FOLDER) or name in ("requirements.txt", "pyproject.toml", "environment.yml"):
                report.add("S4-mlx", os.path.relpath(os.path.join(root, name), folder),
                           "does not belong in a model folder: weights come from the pins, and code runs only as the MLX Swift runner.", "delete it.")


def check_evidence(folder, m, report):
    records = os.path.join(folder, "results/quick-checks")
    tried = set()
    if os.path.isdir(records):
        for name in os.listdir(records):
            try:
                record = json.load(open(os.path.join(records, name)))
            except ValueError:
                continue
            for row in record.get("rows", []):
                tried |= set(row.get("ops") or {})
    for op in resolved_ops(folder, m)[0]:
        if op.get("name") not in tried:
            report.add("S7-tried", f"[[op]] {op.get('name')}", "has no measured run.", f"kit/new-model.py try ID {op.get('name')} '{{…}}' with a real input.")
    readme = os.path.join(folder, "README.md")
    if not os.path.exists(readme):
        report.add("S8-readme", "README.md", "is missing.", "write what the model does, its inputs and its limits.")


def run(folder, kinds_dir=None, hub=None):
    """All static checks. Returns (report, facts)."""
    report = Report()
    hub = hub or os.path.expanduser("~/.cache/huggingface/hub")
    manifest = os.path.join(folder, "model.toml")
    if not os.path.exists(manifest):
        report.add("S1-toml", "model.toml", "is missing.", "scaffold the folder with the kit.")
        return report, {}
    try:
        m = read_toml(manifest)
    except Exception as error:
        report.add("S1-toml", "model.toml", f"does not parse ({error}).", "fix the TOML syntax on that line.")
        return report, {}
    check_manifest(folder, m, report)
    check_licence(m, report)
    check_ops(folder, m, kinds_dir, report)
    facts = check_weights(m, hub, report) or {}
    check_runner(folder, m, report)
    check_evidence(folder, m, report)
    return report, facts
