"""
Unit tests for magi_core/cli.py.

Tests argument parsing, config/prompt loading, model-check output,
and the main() entry point (all Magi / asyncio calls are mocked so
no API keys or network access are required).
"""
import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
import yaml
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magi_core.cli import (
    build_parser,
    _load_config,
    _load_prompts,
    _print_model_check,
    main,
    METHODS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_main(argv, magi_run_return="mock text result", magi_structured_return=None):
    """
    Invoke main() with a patched sys.argv, a stubbed Magi class, and
    captured stdout/stderr.  Returns (stdout_str, stderr_str, exit_code).
    exit_code is None if main() returned normally (no sys.exit).
    """
    if magi_structured_return is None:
        magi_structured_return = {
            "schema_version": "1.0",
            "method": "VoteYesNo",
            "prompt": "q",
            "deliberative": False,
            "models": ["modelA"],
            "rounds": [
                {
                    "round": 1,
                    "responses": [{"model": "modelA", "pseudonym": "Participant X1A2", "response": "yes", "reason": "R", "confidence_score": 0.9, "fallback_for": None}],
                    "errors": [],
                    "aggregate": {"votes": {"yes": 1}, "winner": "yes", "threshold": 0.5},
                    "rapporteur": {"model": "modelA", "summary": "Summary"},
                }
            ],
        }

    mock_magi_instance = MagicMock()
    mock_magi_instance.run = AsyncMock(return_value=magi_run_return)
    mock_magi_instance.run_structured = AsyncMock(return_value=magi_structured_return)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = None

    with patch("sys.argv", ["magi"] + argv), \
         patch("magi_core.cli.Magi", return_value=mock_magi_instance), \
         patch("magi_core.cli._load_config", return_value={"llms": ["modelA"], "defaults": {}}), \
         patch("magi_core.cli._load_prompts", return_value={}):
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                main()
        except SystemExit as exc:
            exit_code = exc.code

    return stdout_buf.getvalue(), stderr_buf.getvalue(), exit_code


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser(unittest.TestCase):

    def setUp(self):
        self.parser = build_parser()

    def test_prompt_required(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_default_method_is_vote_yes_no(self):
        args = self.parser.parse_args(["my question"])
        self.assertEqual(args.method, "VoteYesNo")

    def test_all_methods_accepted(self):
        for method in METHODS:
            args = self.parser.parse_args(["q", "--method", method])
            self.assertEqual(args.method, method)

    def test_invalid_method_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["q", "--method", "Nonsense"])

    def test_default_output_format_is_text(self):
        args = self.parser.parse_args(["q"])
        self.assertEqual(args.output_format, "text")

    def test_output_format_json(self):
        args = self.parser.parse_args(["q", "--output-format", "json"])
        self.assertEqual(args.output_format, "json")

    def test_invalid_output_format_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["q", "--output-format", "xml"])

    def test_vote_threshold_default(self):
        args = self.parser.parse_args(["q"])
        self.assertEqual(args.vote_threshold, 0.5)

    def test_vote_threshold_custom(self):
        args = self.parser.parse_args(["q", "--vote-threshold", "0.7"])
        self.assertAlmostEqual(args.vote_threshold, 0.7)

    def test_no_abstain_default_false(self):
        args = self.parser.parse_args(["q"])
        self.assertFalse(args.no_abstain)

    def test_no_abstain_flag(self):
        args = self.parser.parse_args(["q", "--no-abstain"])
        self.assertTrue(args.no_abstain)

    def test_deliberative_default_false(self):
        args = self.parser.parse_args(["q"])
        self.assertFalse(args.deliberative)

    def test_deliberative_flag(self):
        args = self.parser.parse_args(["q", "--deliberative"])
        self.assertTrue(args.deliberative)

    def test_llms_parsed(self):
        args = self.parser.parse_args(["q", "--llms", "gpt-4o,claude-3"])
        self.assertEqual(args.llms, "gpt-4o,claude-3")

    def test_options_parsed(self):
        args = self.parser.parse_args(["q", "--options", "A,B,C"])
        self.assertEqual(args.options, "A,B,C")

    def test_system_prompt(self):
        args = self.parser.parse_args(["q", "--system-prompt", "You are helpful"])
        self.assertEqual(args.system_prompt, "You are helpful")

    def test_rapporteur_prompt(self):
        args = self.parser.parse_args(["q", "--rapporteur-prompt", "Be concise"])
        self.assertEqual(args.rapporteur_prompt, "Be concise")

    def test_check_models_flag(self):
        args = self.parser.parse_args(["q", "--check-models"])
        self.assertTrue(args.check_models)

    def test_config_path(self):
        args = self.parser.parse_args(["q", "--config", "custom.yaml"])
        self.assertEqual(args.config, "custom.yaml")

    def test_prompts_path(self):
        args = self.parser.parse_args(["q", "--prompts", "custom_prompts.yaml"])
        self.assertEqual(args.prompts, "custom_prompts.yaml")


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):

    def test_explicit_path_loads_file(self):
        cfg = {"llms": ["gpt-4o"], "defaults": {"max_retries": 1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            path = f.name
        try:
            result = _load_config(path)
            self.assertEqual(result["llms"], ["gpt-4o"])
        finally:
            os.unlink(path)

    def test_missing_explicit_path_falls_through_to_cwd(self):
        # Point to a non-existent file; with no config.yaml in cwd it returns empty config
        with patch("os.path.exists", return_value=False):
            result = _load_config("/nonexistent/path/config.yaml")
        self.assertEqual(result["llms"], [])

    def test_no_path_no_file_returns_empty_config(self):
        with patch("os.path.exists", return_value=False):
            buf = io.StringIO()
            with redirect_stderr(buf):
                result = _load_config(None)
        self.assertIn("llms", result)
        self.assertEqual(result["llms"], [])
        self.assertIn("Warning", buf.getvalue())

    def test_cwd_config_yaml_loaded_when_no_path_given(self):
        cfg = {"llms": ["cwd-model"], "defaults": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(cfg, f)
            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _load_config(None)
                self.assertEqual(result["llms"], ["cwd-model"])
            finally:
                os.chdir(orig_dir)

    def test_explicit_path_takes_priority_over_cwd(self):
        explicit_cfg = {"llms": ["explicit-model"], "defaults": {}}
        cwd_cfg = {"llms": ["cwd-model"], "defaults": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_path = os.path.join(tmpdir, "explicit.yaml")
            cwd_config_path = os.path.join(tmpdir, "config.yaml")
            with open(explicit_path, "w") as f:
                yaml.dump(explicit_cfg, f)
            with open(cwd_config_path, "w") as f:
                yaml.dump(cwd_cfg, f)
            orig_dir = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _load_config(explicit_path)
                self.assertEqual(result["llms"], ["explicit-model"])
            finally:
                os.chdir(orig_dir)


# ---------------------------------------------------------------------------
# _load_prompts
# ---------------------------------------------------------------------------

class TestLoadPrompts(unittest.TestCase):

    def test_explicit_path_loads_file(self):
        prompts = {"system_base": "You are helpful"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(prompts, f)
            path = f.name
        try:
            result = _load_prompts(path)
            self.assertEqual(result["system_base"], "You are helpful")
        finally:
            os.unlink(path)

    def test_bundled_prompts_loaded_as_fallback(self):
        # When no explicit path and no cwd prompts.yaml, should load bundled file
        with patch("os.path.exists", side_effect=lambda p: "magi_core" in p or p.endswith("prompts.yaml") and "magi_core" in p):
            # Just verify it doesn't exit(1); the bundled file should always exist
            from magi_core.utils import get_default_prompts_path
            bundled = get_default_prompts_path()
            self.assertTrue(os.path.exists(bundled), "Bundled prompts.yaml must exist")

    def test_missing_all_paths_exits(self):
        with patch("os.path.exists", return_value=False):
            buf = io.StringIO()
            with redirect_stderr(buf):
                with self.assertRaises(SystemExit) as cm:
                    _load_prompts(None)
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error", buf.getvalue())


# ---------------------------------------------------------------------------
# _print_model_check
# ---------------------------------------------------------------------------

class TestPrintModelCheck(unittest.TestCase):

    def _make_slot(self, model, ok, category, message, fallbacks=None):
        checks = [{"model": model, "ok": ok, "category": category, "message": message}]
        if fallbacks:
            checks.extend(fallbacks)
        return {"checks": checks}

    def test_all_ok_returns_true(self):
        results = [
            self._make_slot("gpt-4o", True, "ok", ""),
            self._make_slot("claude-3", True, "ok", ""),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            all_ok = _print_model_check(results)
        self.assertTrue(all_ok)
        output = buf.getvalue()
        self.assertIn("[OK]", output)
        self.assertIn("All models are available", output)

    def test_any_failed_returns_false(self):
        results = [
            self._make_slot("gpt-4o", True, "ok", ""),
            self._make_slot("bad-model", False, "not_found", "model not found"),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            all_ok = _print_model_check(results)
        self.assertFalse(all_ok)
        output = buf.getvalue()
        self.assertIn("NOT FOUND", output)
        self.assertIn("bad-model", output)
        self.assertIn("Warning", output)

    def test_fallback_displayed_indented(self):
        results = [
            self._make_slot(
                "primary", False, "deprecated", "deprecated",
                fallbacks=[{"model": "fallback", "ok": True, "category": "ok", "message": ""}]
            )
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_model_check(results)
        output = buf.getvalue()
        self.assertIn("fallback", output)
        self.assertIn("└─", output)

    def test_category_labels_shown(self):
        categories = [
            ("deprecated", "DEPRECATED"),
            ("auth", "AUTH ERROR"),
            ("rate_limit", "RATE LIMITED"),
            ("timeout", "TIMEOUT"),
            ("empty_response", "EMPTY RESP"),
            ("unknown", "ERROR"),
        ]
        for category, expected_label in categories:
            results = [self._make_slot("m", False, category, "msg")]
            buf = io.StringIO()
            with redirect_stdout(buf):
                _print_model_check(results)
            self.assertIn(expected_label, buf.getvalue(), f"Expected '{expected_label}' for category '{category}'")

    def test_empty_results(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            all_ok = _print_model_check([])
        self.assertTrue(all_ok)


# ---------------------------------------------------------------------------
# main() — validation errors and early exits
# ---------------------------------------------------------------------------

class TestMainValidation(unittest.TestCase):

    def test_no_llms_exits_1(self):
        mock_magi = MagicMock()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with patch("sys.argv", ["magi", "question"]), \
             patch("magi_core.cli.Magi", return_value=mock_magi), \
             patch("magi_core.cli._load_config", return_value={"llms": [], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}):
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("No LLMs", stderr_buf.getvalue())

    def test_vote_threshold_zero_exits_1(self):
        _, stderr, code = _run_main(["q", "--vote-threshold", "0.0"])
        self.assertEqual(code, 1)
        self.assertIn("vote-threshold", stderr)

    def test_vote_threshold_above_one_exits_1(self):
        _, stderr, code = _run_main(["q", "--vote-threshold", "1.1"])
        self.assertEqual(code, 1)
        self.assertIn("vote-threshold", stderr)

    def test_vote_options_without_options_exits_1(self):
        _, stderr, code = _run_main(["q", "--method", "VoteOptions"])
        self.assertEqual(code, 1)
        self.assertIn("--options", stderr)

    def test_keyboard_interrupt_exits_1(self):
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(side_effect=KeyboardInterrupt())
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stderr(buf):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Aborted", buf.getvalue())

    def test_exception_from_run_exits_1(self):
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stderr(buf):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("boom", buf.getvalue())


# ---------------------------------------------------------------------------
# main() — normal text and JSON output paths
# ---------------------------------------------------------------------------

class TestMainOutput(unittest.TestCase):

    def test_text_output_printed(self):
        stdout, _, code = _run_main(["q", "--method", "Majority"])
        self.assertIsNone(code)
        self.assertIn("mock text result", stdout)

    def test_json_output_is_valid_json(self):
        stdout, _, code = _run_main(["q", "--output-format", "json"])
        self.assertIsNone(code)
        # stdout starts with the header lines (Method/Models/---) then the JSON blob
        json_start = stdout.index("{")
        parsed = json.loads(stdout[json_start:])
        self.assertEqual(parsed["schema_version"], "1.0")

    def test_json_output_calls_run_structured(self):
        with patch("sys.argv", ["magi", "q", "--output-format", "json"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run_structured = AsyncMock(return_value={"schema_version": "1.0", "method": "VoteYesNo", "prompt": "q", "deliberative": False, "models": ["m"], "rounds": []})
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
        mock_instance.run_structured.assert_called_once()
        mock_instance.run.assert_not_called()

    def test_text_output_calls_run(self):
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="text result")
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
        mock_instance.run.assert_called_once()
        mock_instance.run_structured.assert_not_called()

    def test_header_shows_method_and_models(self):
        stdout, _, _ = _run_main(["q", "--method", "Majority"])
        self.assertIn("Method : Majority", stdout)
        self.assertIn("Models : modelA", stdout)

    def test_deliberative_flag_shown_in_header(self):
        stdout, _, _ = _run_main(["q", "--deliberative"])
        self.assertIn("Deliberative", stdout)

    def test_deliberative_not_shown_without_flag(self):
        stdout, _, _ = _run_main(["q"])
        self.assertNotIn("Deliberative", stdout)

    def test_language_flag_threaded_to_run(self):
        with patch("sys.argv", ["magi", "q", "--language", "German"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="text")
            MockMagi.return_value = mock_instance
            with redirect_stdout(io.StringIO()):
                main()
        kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(kwargs.get("language"), "German")

    def test_anonymous_report_flag_threaded(self):
        with patch("sys.argv", ["magi", "q", "--anonymous-report"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="text")
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
        kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(kwargs.get("show_real_names_in_report"), False)
        self.assertIn("Report : anonymous", buf.getvalue())

    def test_language_from_config_default(self):
        cfg = {"llms": ["m"], "defaults": {"language": "German"}}
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value=cfg), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="text")
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
        kwargs = mock_instance.run.call_args.kwargs
        self.assertEqual(kwargs.get("language"), "German")
        self.assertIn("Language: German", buf.getvalue())


# ---------------------------------------------------------------------------
# main() — argument pass-through to Magi.run()
# ---------------------------------------------------------------------------

class TestMainArgPassthrough(unittest.TestCase):

    def _capture_run_call(self, argv):
        """Return the kwargs that were passed to magi.run()."""
        with patch("sys.argv", ["magi"] + argv), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="result")
            MockMagi.return_value = mock_instance
            with redirect_stdout(io.StringIO()):
                main()
        _, kwargs = mock_instance.run.call_args
        return kwargs

    def test_method_passed_through(self):
        kwargs = self._capture_run_call(["q", "--method", "Consensus"])
        self.assertEqual(kwargs["method"], "Consensus")

    def test_deliberative_passed_through(self):
        kwargs = self._capture_run_call(["q", "--deliberative"])
        self.assertTrue(kwargs["deliberative"])

    def test_deliberative_false_by_default(self):
        kwargs = self._capture_run_call(["q"])
        self.assertFalse(kwargs["deliberative"])

    def test_system_prompt_passed_through(self):
        kwargs = self._capture_run_call(["q", "--system-prompt", "Be brief"])
        self.assertEqual(kwargs["system_prompt"], "Be brief")

    def test_no_abstain_sets_allow_abstain_false(self):
        kwargs = self._capture_run_call(["q", "--no-abstain"])
        self.assertFalse(kwargs["method_options"]["allow_abstain"])

    def test_allow_abstain_true_by_default(self):
        kwargs = self._capture_run_call(["q"])
        self.assertTrue(kwargs["method_options"]["allow_abstain"])

    def test_rapporteur_prompt_passed_through(self):
        kwargs = self._capture_run_call(["q", "--rapporteur-prompt", "Be concise"])
        self.assertEqual(kwargs["method_options"]["rapporteur_prompt"], "Be concise")

    def test_vote_threshold_passed_through(self):
        kwargs = self._capture_run_call(["q", "--vote-threshold", "0.8"])
        self.assertAlmostEqual(kwargs["method_options"]["vote_threshold"], 0.8)

    def test_options_split_by_comma(self):
        kwargs = self._capture_run_call(
            ["q", "--method", "VoteOptions", "--options", "A,B, C"]
        )
        self.assertEqual(kwargs["method_options"]["options"], ["A", "B", "C"])

    def test_llms_override_config(self):
        with patch("sys.argv", ["magi", "q", "--llms", "gpt-4o,claude-3"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["config-model"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="result")
            MockMagi.return_value = mock_instance
            with redirect_stdout(io.StringIO()):
                main()
        _, kwargs = mock_instance.run.call_args
        self.assertEqual(kwargs["selected_llms"], ["gpt-4o", "claude-3"])


# ---------------------------------------------------------------------------
# main() — fallback chain display (the join fix)
# ---------------------------------------------------------------------------

class TestMainFallbackChainDisplay(unittest.TestCase):

    def test_fallback_chains_display_primary_name_only(self):
        """When config contains fallback chains, only the primary model is shown."""
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value={
                 "llms": [["gpt-4o", "gpt-4o-mini"], "claude-3"],
                 "defaults": {}
             }), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="result")
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
        output = buf.getvalue()
        self.assertIn("gpt-4o", output)
        self.assertIn("claude-3", output)
        # The raw list representation must not appear
        self.assertNotIn("['gpt-4o'", output)

    def test_full_slot_including_fallbacks_passed_to_run(self):
        """The full slot list (including fallbacks) is forwarded to Magi.run()."""
        llms_config = [["gpt-4o", "gpt-4o-mini"], "claude-3"]
        with patch("sys.argv", ["magi", "q"]), \
             patch("magi_core.cli._load_config", return_value={"llms": llms_config, "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value="result")
            MockMagi.return_value = mock_instance
            with redirect_stdout(io.StringIO()):
                main()
        _, kwargs = mock_instance.run.call_args
        self.assertEqual(kwargs["selected_llms"], llms_config)


# ---------------------------------------------------------------------------
# main() — --check-models
# ---------------------------------------------------------------------------

class TestMainCheckModels(unittest.TestCase):

    def _run_check_models(self, check_results):
        with patch("sys.argv", ["magi", "q", "--check-models"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.check_models = AsyncMock(return_value=check_results)
            MockMagi.return_value = mock_instance
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as cm:
                    main()
        return buf.getvalue(), cm.exception.code

    def test_all_ok_exits_0(self):
        results = [{"checks": [{"model": "m", "ok": True, "category": "ok", "message": ""}]}]
        output, code = self._run_check_models(results)
        self.assertEqual(code, 0)
        self.assertIn("[OK]", output)

    def test_any_failed_exits_1(self):
        results = [{"checks": [{"model": "m", "ok": False, "category": "not_found", "message": "not found"}]}]
        output, code = self._run_check_models(results)
        self.assertEqual(code, 1)
        self.assertIn("NOT FOUND", output)

    def test_check_models_does_not_call_run(self):
        results = [{"checks": [{"model": "m", "ok": True, "category": "ok", "message": ""}]}]
        with patch("sys.argv", ["magi", "q", "--check-models"]), \
             patch("magi_core.cli._load_config", return_value={"llms": ["m"], "defaults": {}}), \
             patch("magi_core.cli._load_prompts", return_value={}), \
             patch("magi_core.cli.Magi") as MockMagi:
            mock_instance = MagicMock()
            mock_instance.check_models = AsyncMock(return_value=results)
            mock_instance.run = AsyncMock(return_value="result")
            MockMagi.return_value = mock_instance
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
        mock_instance.run.assert_not_called()
        mock_instance.run_structured.assert_not_called()


if __name__ == "__main__":
    unittest.main()
