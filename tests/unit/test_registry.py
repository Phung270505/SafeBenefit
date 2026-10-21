from __future__ import annotations

import json
import re
import unittest

from src.common.config import METHOD_IDS, load_study_config
from src.common.paths import PROJECT_ROOT
from src.methods.registry import METHOD_ORDER, create_selector


class RegistryTests(unittest.TestCase):
    def test_config_python_shell_and_cpp_use_same_registered_ids(self) -> None:
        config = load_study_config()
        self.assertEqual(tuple(config["active_methods"]), METHOD_IDS)
        self.assertEqual(METHOD_ORDER, METHOD_IDS)
        self.assertEqual(METHOD_IDS, ("STATIC", "SWITCH_COST_THRESHOLD", "LINUCB", "SAFE_BENEFIT"))
        self.assertEqual(config["comparators"], ["STATIC", "SWITCH_COST_THRESHOLD", "LINUCB"])
        self.assertEqual(config["statistics"]["holm_comparisons"], 3)
        cpp = (PROJECT_ROOT / "simulator/omnetpp/graph_query/GraphQueryApp.cc").read_text()
        shell = (PROJECT_ROOT / "simulator/omnetpp/tools/run_case.sh").read_text()
        self.assertEqual(set(re.findall(r'method != "([A-Z_]+)"', cpp)), set(METHOD_IDS))
        shell_methods = re.search(r'\[\[ "\$method" =~ \^\(([^)]+)\)\$', shell)
        self.assertIsNotNone(shell_methods)
        self.assertEqual(tuple(shell_methods.group(1).split("|")), METHOD_IDS)
        for method in METHOD_IDS:
            self.assertIn(method, cpp)
            self.assertIn(method, shell)
            self.assertEqual(create_selector(method).method_id, method)

    def test_retired_methods_cannot_be_selected(self) -> None:
        for method in ("FIXED_THRESHOLD", "TUNED_THRESHOLD", "LEARNED_COST"):
            with self.assertRaisesRegex(ValueError, "unregistered"):
                create_selector(method)

    def test_single_config_is_valid_json(self) -> None:
        payload = json.loads((PROJECT_ROOT / "config/study.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["study_id"], "SAFEBENEFIT-DGQ")
        self.assertEqual(payload["master_seed"], 42)


if __name__ == "__main__":
    unittest.main()
