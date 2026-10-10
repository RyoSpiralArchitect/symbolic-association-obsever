"""Small synthetic vectors here are test inputs, never demo model outputs."""
import importlib.util
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from trace_inspector import (Dataset, HERE, ThreadingHTTPServer, compare_vectors,
                             make_handler, numeric_vector, read_annotations,
                             resolve_state_file, summarize_payload)


class AnnotationTests(unittest.TestCase):
    def test_demo_has_no_invented_measurements(self):
        records = read_annotations(HERE / "examples/moon_demo.jsonl")
        self.assertEqual(len(records), 3)
        self.assertEqual({r["tags"][0] for r in records}, {"literal", "metaphorical", "mythical"})
        for r in records:
            self.assertIsNone(r["state_file"])
            self.assertNotIn("token_id", r)
            self.assertEqual(r["fixture_kind"], "handwritten_text_only")
        self.assertEqual(Dataset(records, demo=True).trace(0)["hidden"], [])

    def test_empty_blank_and_unicode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.jsonl"
            path.write_text('\n{"level":"segment","segment_text":"月","tags":[]}\n\n', encoding="utf-8")
            self.assertEqual(read_annotations(path)[0]["segment_text"], "月")
            path.write_text("")
            self.assertEqual(read_annotations(path), [])

    def test_invalid_records_and_json(self):
        bad = ["{bad}", "[]", '{"level":"unknown"}', '{"level":"token","tags":"mythic"}',
               '{"level":"token","state_file":2}', '{"level":"token","note":null}']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.jsonl"
            for text in bad:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_annotations(path)


class StateTests(unittest.TestCase):
    def test_root_confinement_and_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "capture"
            root.mkdir()
            state = root / "file.pt"
            state.touch()
            outside = Path(folder) / "outside.pt"
            outside.touch()
            self.assertEqual(resolve_state_file("file.pt", root), state)
            self.assertEqual(resolve_state_file(str(state), root), state)
            for path in ["../outside.pt", str(outside)]:
                with self.assertRaises(ValueError):
                    resolve_state_file(path, root)
            (root / "link.pt").symlink_to(outside)
            with self.assertRaises(ValueError):
                resolve_state_file("link.pt", root)
            with self.assertRaises(ValueError):
                resolve_state_file("file.pt", None)
            with self.assertRaises(ValueError):
                resolve_state_file("missing.pt", root)

    def test_summaries_token_and_segment(self):
        summary = summarize_payload({"hidden_states": [[3, 4], [1, 0]],
            "attentions_from_token": [[[0.2, 0.8], [0.4, 0.6]], None]}, "token")
        self.assertEqual(summary["hidden"][0]["norm"], 5)
        self.assertEqual(summary["attention"][0]["top_positions"][0]["position"], 1)
        self.assertAlmostEqual(summary["attention"][0]["top_positions"][0]["weight"], 0.7)
        self.assertTrue(summary["attention"][1]["missing"])
        segment = summarize_payload({"hidden_states_segment_mean": [[3, 4]]}, "segment")
        self.assertEqual(segment["hidden"][0]["norm"], 5)
        self.assertEqual(summarize_payload({"attentions_from_token": [[[1]]]}, "token")["hidden"], [])

    def test_invalid_shapes_and_values(self):
        for vector in [[], [float("nan")], [float("inf")], [True], ["3"], [[1]], [1e308, 1e308, 1e308, 1e308]]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                numeric_vector(vector)
        for payload in [{"hidden_states": [[1], [1, 2]]}, {"hidden_states": None},
                        {"attentions_from_token": [[[1], [1, 2]]]}, {"attentions_from_token": [[1, 2]]}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                summarize_payload(payload, "token")

    def test_no_state_does_not_require_torch(self):
        with patch("trace_inspector.importlib.import_module", side_effect=AssertionError("Do not import torch")):
            self.assertEqual(Dataset([{"level": "token", "state_file": None}]).trace(0)["status"], "unavailable")
            disabled = Dataset([{"level": "token", "state_file": "states/a.pt"}]).trace(0)
            self.assertIn("loading is off", disabled["message"])

    def test_missing_torch_has_helpful_message(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "a.pt").touch()
            with patch("trace_inspector.importlib.import_module", side_effect=ImportError()):
                result = Dataset([{"level": "token", "state_file": "a.pt"}], folder).trace(0)
            self.assertIn("PyTorch is required", result["message"])

    def test_safe_loader_arguments_and_tensor_conversion(self):
        class Tensor:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def tolist(self): return [3.0, 4.0]
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "a.pt"
            state.touch()
            from unittest.mock import Mock
            loader = Mock(return_value={"hidden_states": [Tensor()]})
            fake_torch = SimpleNamespace(Tensor=Tensor, load=loader)
            with patch("trace_inspector.importlib.import_module", return_value=fake_torch):
                dataset = Dataset([{"level": "token", "state_file": "a.pt"}], folder)
                self.assertEqual(dataset.trace(0)["hidden"][0]["norm"], 5)
                self.assertEqual(dataset.trace(0)["status"], "available")
            loader.assert_called_once_with(state, map_location="cpu", weights_only=True)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed; real .pt round-trip not run")
    def test_real_torch_round_trip(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            torch.save({"hidden_states": [torch.tensor([3., 4.])],
                        "attentions_from_token": [torch.tensor([[0.2, 0.8]])]}, Path(folder) / "a.pt")
            result = Dataset([{"level": "token", "state_file": "a.pt"}], folder).trace(0)
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["hidden"][0]["norm"], 5)
            self.assertEqual(result["attention"][0]["sequence_length"], 2)


class ComparisonTests(unittest.TestCase):
    def test_known_geometry_and_zero_norm(self):
        result = compare_vectors([[1, 0], [0, 0], [1, 0]], [[0, 1], [1, 0], [-1, 0]])
        self.assertEqual(result[0]["cosine"], 0)
        self.assertAlmostEqual(result[0]["distance"], math.sqrt(2))
        self.assertIsNone(result[1]["cosine"])
        self.assertEqual(result[2]["cosine"], -1)
        self.assertAlmostEqual(compare_vectors([[1e200]], [[1e200]])[0]["cosine"], 1)

    def test_incompatible_data(self):
        for left, right in [([], []), ([[1]], [[1], [1]]), ([[1]], [[1, 2]])]:
            with self.assertRaises(ValueError):
                compare_vectors(left, right)
        with self.assertRaisesRegex(ValueError, "Token and segment"):
            Dataset([{"level": "token"}, {"level": "segment"}]).compare(0, 1)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(read_annotations(HERE / "examples/moon_demo.jsonl"), demo=True)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.dataset))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_endpoints_and_no_vector_leak(self):
        with urlopen(self.url + "/api/annotations") as response:
            self.assertEqual(len(json.load(response)["records"]), 3)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        with urlopen(self.url + "/api/trace?id=0") as response:
            self.assertNotIn("vectors", json.load(response))
        for path in ["/", "/inspector.js", "/inspector.css"]:
            with urlopen(self.url + path) as response:
                self.assertEqual(response.status, 200)

    def test_invalid_requests_and_host(self):
        for path, status in [("/api/trace?id=-1", 400), ("/api/trace?id=999", 400),
                             ("/api/trace?id=abc", 400), ("/api/trace", 400),
                             ("/api/compare?left=0&right=1", 400), ("/symbolic_viewer.py", 404),
                             ("/../README.md", 404)]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as caught:
                urlopen(self.url + path)
            self.assertEqual(caught.exception.code, status)
        with self.assertRaises(HTTPError) as caught:
            urlopen(Request(self.url + "/api/annotations", headers={"Host": "evil.example"}))
        self.assertEqual(caught.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
