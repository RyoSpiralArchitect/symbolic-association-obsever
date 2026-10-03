"""Read-only, localhost inspector for SymbolicViewer JSONL and optional .pt exports."""

import argparse
import importlib
import json
import math
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
MAX_RECORDS = 1000
MAX_JSONL_BYTES = 10 * 1024 * 1024
MAX_STATE_BYTES = 128 * 1024 * 1024


def read_annotations(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSONL_BYTES:
        raise ValueError("JSONL exceeds the 10 MiB limit; use a smaller selection.")
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
        if not isinstance(record, dict) or record.get("level") not in ("token", "segment"):
            raise ValueError(f"Line {line_no}: expected a token or segment annotation.")
        for key in ("prompt", "generated_text", "decoded", "normalized", "segment_text", "note"):
            if key in record and not isinstance(record[key], str):
                raise ValueError(f"Line {line_no}: {key} must be text.")
        if not isinstance(record.get("tags", []), list) or not all(
            isinstance(tag, str) for tag in record.get("tags", [])
        ):
            raise ValueError(f"Line {line_no}: tags must be a list of strings.")
        if record.get("state_file") is not None and not isinstance(record["state_file"], str):
            raise ValueError(f"Line {line_no}: state_file must be a path or null.")
        records.append(record)
        if len(records) > MAX_RECORDS:
            raise ValueError(f"At most {MAX_RECORDS} annotations can be inspected at once.")
    return records


def resolve_state_file(state_file, states_root):
    """Paths in exports are relative to the captor's working directory, not JSONL."""
    if states_root is None:
        raise ValueError("Tensor loading is off. Restart with --states-root pointing to the capture working directory.")
    root = Path(states_root).resolve()
    candidate = Path(state_file)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("State path leaves --states-root; it was not opened.")
    if candidate.suffix != ".pt":
        raise ValueError("Only .pt state exports are supported.")
    if not candidate.is_file():
        raise ValueError("Saved state file was not found under --states-root.")
    if candidate.stat().st_size > MAX_STATE_BYTES:
        raise ValueError("State file exceeds 128 MiB; use a smaller export.")
    return candidate


def numeric_vector(value):
    if not isinstance(value, list) or not value or len(value) > 65536:
        raise ValueError("Expected a nonempty hidden vector of at most 65,536 dimensions.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("State contains a nonnumeric or non-finite hidden value.")
    # Normalize values to Python float so metric arithmetic cannot overflow integer conversion later.
    result = [float(x) for x in value]
    if not math.isfinite(math.hypot(*result)):
        raise ValueError("Hidden-vector norm is not finite.")
    return result


def summarize_payload(payload, level):
    """Input is a weights-only tensor dictionary, converted to lists by load_trace."""
    if not isinstance(payload, dict):
        raise ValueError("Expected the captor's state dictionary.")
    hidden_key = "hidden_states" if level == "token" else "hidden_states_segment_mean"
    attention_key = "attentions_from_token" if level == "token" else "attentions_from_segment_mean"
    hidden = payload.get(hidden_key, [])
    attention = payload.get(attention_key, [])
    if not isinstance(hidden, list) or not isinstance(attention, list) or max(len(hidden), len(attention)) > 256:
        raise ValueError("Expected at most 256 stored layer slots.")
    vectors = [numeric_vector(vector) for vector in hidden]
    if vectors and len({len(vector) for vector in vectors}) != 1:
        raise ValueError("Hidden dimensions differ between stored layers.")
    summaries = []
    for slot, heads in enumerate(attention):
        if heads is None:
            summaries.append({"slot": slot, "missing": True})
            continue
        if not isinstance(heads, list) or not heads or len(heads) > 1024:
            raise ValueError("Expected attention with shape [heads, full_sequence].")
        rows = [numeric_vector(head) for head in heads]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("Attention heads have different sequence lengths.")
        means = [math.fsum(row[i] / len(rows) for row in rows) for i in range(len(rows[0]))]
        if not all(math.isfinite(value) for value in means):
            raise ValueError("Attention means are not finite.")
        top = sorted(enumerate(means), key=lambda item: (-item[1], item[0]))[:5]
        summaries.append({"slot": slot, "heads": len(rows), "sequence_length": len(means),
                          "top_positions": [{"position": pos, "weight": weight} for pos, weight in top]})
    return {"vectors": vectors,
            "hidden": [{"index": i, "dimensions": len(v), "norm": math.hypot(*v)} for i, v in enumerate(vectors)],
            "attention": summaries}


def compare_vectors(left, right):
    if not left or not right:
        raise ValueError("Both annotations need saved hidden states for a comparison.")
    if len(left) != len(right):
        raise ValueError("Stored hidden-layer counts differ; comparison is unavailable.")
    result = []
    for index, (a, b) in enumerate(zip(left, right)):
        if len(a) != len(b):
            raise ValueError("Hidden dimensions differ; comparison is unavailable.")
        norm_a, norm_b = math.hypot(*a), math.hypot(*b)
        cosine = (max(-1.0, min(1.0, math.fsum((x / norm_a) * (y / norm_b) for x, y in zip(a, b))))
                  if norm_a and norm_b else None)
        distance = math.hypot(*(x - y for x, y in zip(a, b)))
        if not math.isfinite(distance):
            raise ValueError("Distance is not finite; comparison is unavailable.")
        result.append({"index": index, "cosine": cosine, "distance": distance, "norm_a": norm_a, "norm_b": norm_b})
    return result


class Dataset:
    def __init__(self, records, states_root=None, demo=False):
        self.records, self.states_root, self.demo = records, states_root, demo

    def public(self):
        return {"demo": self.demo, "records": [dict(record, inspector_id=i) for i, record in enumerate(self.records)]}

    @lru_cache(maxsize=16)
    def trace(self, index):
        record = self.records[index]
        empty = {"vectors": [], "hidden": [], "attention": []}
        if self.demo:
            return dict(empty, status="unavailable", message="Handwritten demo: no model was run and no tensors exist.")
        if not record.get("state_file"):
            return dict(empty, status="unavailable", message="This annotation has no saved state file.")
        try:
            path = resolve_state_file(record["state_file"], self.states_root)
            try:
                torch = importlib.import_module("torch")
            except ImportError as exc:
                raise ValueError("PyTorch is required for .pt files. Use your capture environment; text browsing still works.") from exc
            # Never fall back to unrestricted pickle. Only open locally trusted exports.
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("Expected the captor's state dictionary.")
            keys = ("hidden_states", "hidden_states_segment_mean", "attentions_from_token", "attentions_from_segment_mean")
            converted = {}
            for key in keys:
                if key not in payload:
                    continue
                if not isinstance(payload[key], (list, tuple)):
                    raise ValueError(f"{key} must contain a list of tensor layers.")
                converted[key] = []
                for tensor in payload[key]:
                    if tensor is None and key.startswith("attentions"):
                        converted[key].append(None)
                    elif isinstance(tensor, torch.Tensor):
                        converted[key].append(tensor.detach().float().cpu().tolist())
                    else:
                        raise ValueError(f"{key} contains a non-tensor layer.")
            result = summarize_payload(converted, record["level"])
            if not result["hidden"] and not result["attention"]:
                raise ValueError("No tensor keys matched this annotation level.")
            return dict(result, status="available", message="Saved post-generation re-forward tensors")
        except Exception as exc:
            # A single missing/corrupt export must not hide the remaining annotations.
            return dict(empty, status="unavailable", message=f"Could not read state: {exc}")

    def compare(self, left, right):
        if self.records[left]["level"] != self.records[right]["level"]:
            raise ValueError("Token and segment-mean representations cannot be compared here.")
        return compare_vectors(self.trace(left)["vectors"], self.trace(right)["vectors"])


def make_handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Guard local-data responses against DNS rebinding; do not enable CORS.
            allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed_hosts:
                self.send_error(403)
                return
            request = urlsplit(self.path)
            try:
                if request.path in ("/", "/inspector.js", "/inspector.css"):
                    filename = "index.html" if request.path == "/" else request.path[1:]
                    mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[filename.rsplit(".", 1)[1]]
                    self.respond((HERE / "inspector" / filename).read_bytes(), mime)
                elif request.path == "/api/annotations":
                    self.respond(dataset.public())
                elif request.path == "/api/trace":
                    index = self.record_id(parse_qs(request.query), "id")
                    result = dict(dataset.trace(index))
                    result.pop("vectors")
                    self.respond(result)
                elif request.path == "/api/compare":
                    query = parse_qs(request.query)
                    self.respond({"layers": dataset.compare(self.record_id(query, "left"), self.record_id(query, "right"))})
                else:
                    self.send_error(404)
            except (ValueError, IndexError, KeyError) as exc:
                self.respond({"error": str(exc)}, status=400)

        def record_id(self, query, key):
            index = int(query[key][0])
            if not 0 <= index < len(dataset.records):
                raise ValueError("Unknown annotation.")
            return index

        def respond(self, data, mime="application/json", status=200):
            content = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(content)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Handwritten moon examples; no model/tensors required")
    source.add_argument("--annotations", type=Path, help="JSONL written by symbolic_viewer.py")
    parser.add_argument("--states-root", type=Path, help="Opt in to loading trusted .pt exports below this capture working directory")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        records = read_annotations(HERE / "examples" / "moon_demo.jsonl" if args.demo else args.annotations)
        dataset = Dataset(records, args.states_root, args.demo)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(dataset))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Inspector could not start: {exc}\n")
    print(f"Symbolic Trace Inspector → http://127.0.0.1:{server.server_port}", flush=True)
    print("Read-only, local only. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
