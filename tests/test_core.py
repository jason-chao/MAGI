import unittest
import asyncio
import json
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magi_core.core import Magi

# ---------------------------------------------------------------------------
# Moral-dilemma prompts used across tests (mirrors README examples)
# ---------------------------------------------------------------------------
# Autonomous vehicle trolley problem — the AI must be pre-programmed to kill
# someone; inaction is also a choice. Harder than the classic lever variant.
TROLLEY_PROBLEM = (
    "A self-driving car's brakes have failed. It will kill five pedestrians "
    "unless its AI deliberately swerves onto the pavement, killing the single "
    "passenger inside. Should the AI be programmed to sacrifice its passenger?"
)

# Capital punishment + epistemic humility: even if the crime warrants death,
# wrongful convictions are inevitable — does that change the calculus?
CAPITAL_PUNISHMENT = (
    "Should capital punishment be re-introduced for terrorist attacks that cause "
    "mass civilian casualties, even knowing that wrongful convictions are "
    "statistically inevitable?"
)

# Lifeboat ethics: incommensurable claims on a single heart.
HEART_TRANSPLANT = "A hospital has one donor heart. Who should receive it?"

# Repugnant conclusion: no comfortable resting place for any ethical theory.
REPUGNANT_CONCLUSION = (
    "Derek Parfit's Repugnant Conclusion holds that a trillion people living "
    "lives barely worth living is morally preferable to ten billion living "
    "very happy lives, because total well-being is greater. "
    "Is this conclusion repugnant, unavoidable, or does it expose a flaw in "
    "utilitarian reasoning itself?"
)

# Effective altruism's demanding conclusion.
DROWNING_CHILD = (
    "Anyone who spends money on luxuries while children die of preventable "
    "diseases is morally equivalent to letting a drowning child die. "
    "Is this argument sound?"
)

# ---------------------------------------------------------------------------
# Mock prompt configuration (mirrors prompts.yaml structure)
# ---------------------------------------------------------------------------
MOCK_PROMPTS = {
    'system_base': 'System Base Prompt',
    'deliberative_instruction': 'Deliberative Instruction {peer_responses} {original_prompt}',
    'methods': {
        'VoteYesNo': {
            'instruction': 'Vote {options} {prompt}',
            'default_options': ['yes', 'no', 'abstain']
        },
        'VoteOptions': {
            'instruction': 'VoteOptions {options} {prompt}'
        },
        'Majority': {
            'instruction': 'Majority {prompt}'
        },
        'Consensus': {
            'instruction': 'Consensus {prompt}'
        },
        'Minority': {
            'instruction': 'Minority {prompt}'
        },
        'Probability': {
            'instruction': 'Probability {prompt} {abstain_instruction}'
        },
        'Compose': {
            'instruction': 'Compose {prompt}',
            'rating_instruction': 'Rate {candidates} {prompt}'
        },
        'Synthesis': {
            'instruction': 'Synthesis {prompt}'
        }
    },
    'rapporteur_template': 'Rapporteur {context} {result_line} {response_type} {responses} {instruction} {footer}',
    'rapporteur_footer': 'Footer',
    'rapporteur': {
        'VoteYesNo': {
            'context': 'Ctx',
            'result_line': 'Res {result}',
            'response_type': 'reasons',
            'instruction': 'Instr'
        },
        'VoteOptions': {
            'context': 'Ctx',
            'result_line': 'Res {result}',
            'response_type': 'reasons',
            'instruction': 'Instr'
        },
        'Majority': {
            'context': 'Ctx',
            'result_line': '',
            'response_type': 'views',
            'instruction': 'Instr'
        },
        'Consensus': {
            'context': 'Ctx',
            'result_line': '',
            'response_type': 'views',
            'instruction': 'Instr'
        },
        'Minority': {
            'context': 'Ctx',
            'result_line': '',
            'response_type': 'views',
            'instruction': 'Instr'
        },
        'Probability': {
            'context': 'Ctx',
            'result_line': 'Res {result}',
            'response_type': 'analyses',
            'instruction': 'Instr'
        },
        'Synthesis': {
            'context': 'Ctx',
            'result_line': '',
            'response_type': 'views',
            'instruction': 'Instr'
        }
    }
}


class TestMagi(unittest.TestCase):
    def setUp(self):
        # min_models: 1 so that single-model-failure tests still produce a report
        self.config = {'llms': ['modelA', 'modelB'], 'defaults': {'max_retries': 0, 'min_models': 1}}
        self.magi = Magi(self.config, MOCK_PROMPTS)

        self.patcher1 = patch('magi_core.core.litellm.get_llm_provider')
        self.patcher2 = patch('magi_core.core.litellm.validate_environment')
        self.mock_get_provider = self.patcher1.start()
        self.mock_validate_env = self.patcher2.start()
        self.mock_validate_env.return_value = {'keys_in_environment': True}

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()

    def create_mock_completion(self, content):
        m = MagicMock()
        m.choices = [MagicMock()]
        m.choices[0].message.content = content
        return m

    # ------------------------------------------------------------------
    # VoteYesNo
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_vote_yes_no(self, mock_acompletion):
        """VoteYesNo on the autonomous-vehicle dilemma — split vote produces No Majority."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            user_msg = messages[-1]['content']
            if "Rapporteur" in user_msg:
                return self.create_mock_completion("Summary of votes")
            elif "Vote" in user_msg:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "yes", "reason": "Utilitarian: save more lives", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "no", "reason": "Deontological: cannot program an AI to kill its owner", "confidence_score": 0.8}')
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))
        self.assertIn("Result: No Majority", result)
        self.assertIn("Summary of votes", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_vote_yes_no_threshold(self, mock_acompletion):
        """A low threshold on the autonomous-vehicle dilemma can yield a winner."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "Vote" in messages[-1]['content']:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "yes", "reason": "Net lives saved", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "no", "reason": "Cannot betray the passenger", "confidence_score": 0.8}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        opts = {'vote_threshold': 0.4}
        result = asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo", method_options=opts))
        self.assertTrue("Result: yes" in result or "Result: no" in result)

    # ------------------------------------------------------------------
    # VoteOptions
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_vote_options(self, mock_acompletion):
        """VoteOptions on heart transplant allocation — unanimous vote produces a winner."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "VoteOptions" in messages[-1]['content']:
                return self.create_mock_completion('{"response": "A 45-year-old surgeon who saves hundreds of lives per year", "reason": "Highest aggregate benefit", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        opts = {'options': [
            'A 10-year-old child with decades ahead',
            'A 45-year-old surgeon who saves hundreds of lives per year',
            'The patient who has waited longest on the list',
            'Whoever has the highest chance of survival post-transplant',
        ]}
        result = asyncio.run(self.magi.run(HEART_TRANSPLANT, method="VoteOptions", method_options=opts))
        self.assertIn("surgeon", result.lower())

    # ------------------------------------------------------------------
    # Majority
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_majority(self, mock_acompletion):
        """Majority on capital punishment — highest-confidence model writes the summary."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Majority" in content:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "Oppose", "reason": "Irreversibility of wrongful execution outweighs retribution", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "Support", "reason": "Justified retribution for mass atrocity", "confidence_score": 0.7}')
            if kwargs.get('model') == 'modelA':
                return self.create_mock_completion("Majority Summary by A")
            return self.create_mock_completion("Wrong Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(CAPITAL_PUNISHMENT, method="Majority"))
        self.assertIn("Majority Summary by A", result)

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_consensus(self, mock_acompletion):
        """Consensus on abortion — attempts to find common ground on the hardest case."""
        async def smart_side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "Consensus" in messages[-1]['content']:
                return self.create_mock_completion('{"response": "Bodily autonomy is paramount, though later-term cases require additional justification", "reason": "Partial common ground", "confidence_score": 0.8}')
            else:
                return self.create_mock_completion("Consensus Summary")

        mock_acompletion.side_effect = smart_side_effect
        result = asyncio.run(self.magi.run(
            "At what point, if any, does terminating a pregnancy become morally impermissible, and who has the authority to enforce that line?",
            method="Consensus"
        ))
        self.assertIn("Consensus Summary", result)

    # ------------------------------------------------------------------
    # Minority
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_minority(self, mock_acompletion):
        """Minority report on the drowning-child EA argument surfaces overlooked views."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Minority" in content:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "Sound argument", "reason": "Singer\'s logic is valid; distance is morally irrelevant", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "Unsound", "reason": "Demandingness objection: morality cannot require us to give until we are impoverished", "confidence_score": 0.5}')
            if kwargs.get('model') == 'modelB':
                return self.create_mock_completion("Minority Report by B")
            return self.create_mock_completion("Wrong Report")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(DROWNING_CHILD, method="Minority"))
        self.assertIn("Minority Report by B", result)

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_probability(self, mock_acompletion):
        """Probability on moral luck — one model commits, one abstains."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "Probability" in messages[-1]['content']:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "Largely true", "reason": "Resultant luck should not affect blame; only control matters", "confidence_score": 0.8}')
                else:
                    return self.create_mock_completion('{"response": "Contested", "reason": "Intuitions on moral luck are deeply divided; cannot assign probability", "confidence_score": -1.0}')
            return self.create_mock_completion("Prob Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(REPUGNANT_CONCLUSION, method="Probability"))
        self.assertIn("Average: 0.80", result)
        self.assertIn("Prob Summary", result)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_compose(self, mock_acompletion):
        """Compose generates and peer-reviews steel-man arguments for open borders."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Compose" in content:
                return self.create_mock_completion('{"response": "Steel-man argument text", "reason": "Forceful and well-structured", "confidence_score": 1.0}')
            elif "Rate" in content:
                import re
                candidates = re.findall(r"--- Candidate (.*?) ---", content)
                ratings = {c.strip(): {"score": 8.0, "justification": "Compelling"} for c in candidates}
                return self.create_mock_completion(json.dumps({
                    "response": ratings,
                    "reason": "Rated",
                    "confidence_score": 1.0
                }))
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(
            "Write the strongest possible moral argument for the claim that wealthy nations have an absolute obligation to accept unlimited refugees.",
            method="Compose"
        ))
        self.assertIn("Compose Results", result)
        self.assertIn("Rank 1:", result)
        self.assertIn("Steel-man argument text", result)
        self.assertIn("Average Score: 8.00", result)

    # ------------------------------------------------------------------
    # Language directive
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_language_directive_injected_into_system_prompt(self, mock_acompletion):
        """When `language` is set, every agent receives the directive in its system prompt."""
        captured_system_prompts = []

        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            captured_system_prompts.append(messages[0]['content'])
            return self.create_mock_completion(
                '{"response": "yes", "reason": "R", "confidence_score": 0.9}'
            )

        mock_acompletion.side_effect = side_effect
        asyncio.run(self.magi.run_structured(
            TROLLEY_PROBLEM, method="VoteYesNo", language="German",
        ))
        agent_prompts = [p for p in captured_system_prompts if "System Base Prompt" in p]
        self.assertTrue(agent_prompts, "expected at least one agent system prompt")
        for p in agent_prompts:
            self.assertIn("Respond in German.", p)

    @patch('magi_core.core.litellm.acompletion')
    def test_language_none_leaves_system_prompt_unchanged(self, mock_acompletion):
        """Without `language`, no language directive appears in the system prompt."""
        captured_system_prompts = []

        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            captured_system_prompts.append(messages[0]['content'])
            return self.create_mock_completion(
                '{"response": "yes", "reason": "R", "confidence_score": 0.9}'
            )

        mock_acompletion.side_effect = side_effect
        asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        for p in captured_system_prompts:
            self.assertNotIn("Respond in", p)

    # ------------------------------------------------------------------
    # Compose — peer-rating attribution robustness
    # ------------------------------------------------------------------
    def _compose_attribution_side_effect(self, key_transform):
        """Build a litellm side-effect for Compose where reviewers transform
        each candidate's pseudonym into the JSON key via `key_transform(pseudo)`.
        """
        import re

        def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Compose" in content:
                return self.create_mock_completion(
                    '{"response": "draft", "reason": "R", "confidence_score": 1.0}'
                )
            if "Rate" in content:
                pseudos = re.findall(r"--- Candidate (Participant [A-Z0-9]+) ---", content)
                ratings = key_transform(pseudos)
                return self.create_mock_completion(json.dumps({
                    "response": ratings, "reason": "ok", "confidence_score": 1.0,
                }))
            return self.create_mock_completion('{}')
        return side_effect

    def _run_compose_structured(self):
        return asyncio.run(self.magi.run_structured(
            "prompt text", method="Compose",
        ))

    @patch('magi_core.core.litellm.acompletion')
    def test_compose_attribution_id_only_keys(self, mock_acompletion):
        """Reviewer returns bare 4-char IDs as keys; all candidates still get scored."""
        def xform(pseudos):
            return {p.split()[1]: {"score": 7.0, "justification": "j"} for p in pseudos}
        mock_acompletion.side_effect = self._compose_attribution_side_effect(xform)
        result = self._run_compose_structured()
        ranked = result["rounds"][0]["aggregate"]["ranked_candidates"]
        for item in ranked:
            self.assertEqual(item["average_score"], 7.0)
            self.assertEqual(len(item["peer_reviews"]), 2)

    @patch('magi_core.core.litellm.acompletion')
    def test_compose_attribution_prefixed_keys(self, mock_acompletion):
        """Case/punctuation-varied keys (e.g. 'candidate_participant_x7k2') resolve."""
        def xform(pseudos):
            out = {}
            for i, p in enumerate(pseudos):
                pid = p.split()[1]
                key = f"candidate_participant_{pid.lower()}" if i % 2 == 0 else f"Candidate-{pid}"
                out[key] = {"score": 8.0, "justification": "j"}
            return out
        mock_acompletion.side_effect = self._compose_attribution_side_effect(xform)
        result = self._run_compose_structured()
        ranked = result["rounds"][0]["aggregate"]["ranked_candidates"]
        for item in ranked:
            self.assertEqual(item["average_score"], 8.0)
            self.assertEqual(len(item["peer_reviews"]), 2)

    @patch('magi_core.core.litellm.acompletion')
    def test_compose_attribution_list_response(self, mock_acompletion):
        """Reviewer returns a JSON list in candidate order; positional attribution applies."""
        def xform(pseudos):
            return [{"score": 6.0 + i, "justification": "j"} for i, _ in enumerate(pseudos)]
        mock_acompletion.side_effect = self._compose_attribution_side_effect(xform)
        result = self._run_compose_structured()
        ranked = result["rounds"][0]["aggregate"]["ranked_candidates"]
        for item in ranked:
            self.assertTrue(item["peer_reviews"], "expected positional reviews")
            for pr in item["peer_reviews"]:
                self.assertEqual(pr.get("attribution"), "positional")

    @patch('magi_core.core.litellm.acompletion')
    def test_compose_attribution_total_miss_falls_back_positionally(self, mock_acompletion):
        """Dict with unidentifiable keys but matching arity falls back to positional order."""
        def xform(pseudos):
            return {f"Option {chr(ord('A') + i)}": {"score": 5.0, "justification": "j"}
                    for i, _ in enumerate(pseudos)}
        mock_acompletion.side_effect = self._compose_attribution_side_effect(xform)
        result = self._run_compose_structured()
        agg = result["rounds"][0]["aggregate"]
        for item in agg["ranked_candidates"]:
            self.assertEqual(item["average_score"], 5.0)
            self.assertTrue(all(pr.get("attribution") == "positional"
                                for pr in item["peer_reviews"]))
        self.assertNotIn("unattributed_reviewers", agg)

    @patch('magi_core.core.litellm.acompletion')
    def test_compose_unattributed_surfaced_when_arity_mismatches(self, mock_acompletion):
        """Unresolvable keys with mismatched arity are surfaced, not silently zeroed."""
        def xform(pseudos):
            return {"only_one_entry": {"score": 9.0, "justification": "j"}}
        mock_acompletion.side_effect = self._compose_attribution_side_effect(xform)
        result = self._run_compose_structured()
        agg = result["rounds"][0]["aggregate"]
        self.assertIn("unattributed_reviewers", agg)
        self.assertEqual(len(agg["unattributed_reviewers"]), 2)
        for item in agg["ranked_candidates"]:
            self.assertEqual(item["average_score"], 0.0)
            self.assertEqual(item["peer_reviews"], [])

    # ------------------------------------------------------------------
    # Synthesis (new mode)
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_synthesis(self, mock_acompletion):
        """Synthesis on Parfit's Repugnant Conclusion produces a unified response."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Synthesis" in content:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "The conclusion is unavoidable given total utilitarianism", "reason": "Internal logic is valid; reject the premise instead", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "The conclusion reveals utilitarianism is flawed", "reason": "Average or person-affecting views avoid it", "confidence_score": 0.7}')
            # Rapporteur raw response
            if kwargs.get('model') == 'modelA':
                return self.create_mock_completion("Comprehensive Synthesis by A")
            return self.create_mock_completion("Wrong Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(REPUGNANT_CONCLUSION, method="Synthesis"))
        self.assertIn("Synthesis (by modelA)", result)
        self.assertIn("Comprehensive Synthesis by A", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_synthesis_includes_all_views(self, mock_acompletion):
        """Synthesis rapporteur is called with all participant responses."""
        captured_prompts = []

        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            captured_prompts.append(content)
            if "Synthesis" in content:
                return self.create_mock_completion('{"response": "View", "reason": "R", "confidence_score": 0.8}')
            return self.create_mock_completion("Synthesis narrative")

        mock_acompletion.side_effect = side_effect
        asyncio.run(self.magi.run(REPUGNANT_CONCLUSION, method="Synthesis"))

        rapporteur_call = next((p for p in captured_prompts if "Rapporteur" in p), None)
        self.assertIsNotNone(rapporteur_call, "Rapporteur should have been called")

    # ------------------------------------------------------------------
    # Deliberative mode
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_deliberative_mode(self, mock_acompletion):
        """Deliberative round on capital punishment produces pre- and post-deliberation sections."""
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Vote" in content and "Deliberative Instruction" not in content:
                return self.create_mock_completion('{"response": "yes", "reason": "Justified retribution for mass atrocity", "confidence_score": 0.9}')
            elif "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            elif "Deliberative Instruction" in content:
                return self.create_mock_completion('{"response": "no", "reason": "Reconsidered: wrongful executions are irreversible", "confidence_score": 0.95}')
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(CAPITAL_PUNISHMENT, method="VoteYesNo", deliberative=True))
        self.assertIn("Pre-Deliberation Results", result)
        self.assertIn("Post-Deliberation Results", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_synthesis_deliberative(self, mock_acompletion):
        """Synthesis also supports the deliberative round."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Synthesis" in content:
                return self.create_mock_completion('{"response": "View", "reason": "R", "confidence_score": 0.8}')
            return self.create_mock_completion("Synthesis narrative")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(REPUGNANT_CONCLUSION, method="Synthesis", deliberative=True))
        self.assertIn("Pre-Deliberation", result)
        self.assertIn("Post-Deliberation", result)

    # ------------------------------------------------------------------
    # No-abstain / rapporteur prompt
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_no_abstain(self, mock_acompletion):
        """no_abstain flag appends the disallow instruction to the prompt."""
        mock_acompletion.return_value = self.create_mock_completion('{"response": "yes", "confidence_score": 0.9}')
        opts = {'allow_abstain': False}
        asyncio.run(self.magi.run(CAPITAL_PUNISHMENT, method="VoteYesNo", method_options=opts))
        found = any(
            "Abstaining is not allowed" in call[1].get('messages', [{}])[-1].get('content', '')
            for call in mock_acompletion.call_args_list
        )
        self.assertTrue(found)

    @patch('magi_core.core.litellm.acompletion')
    def test_custom_rapporteur_prompt(self, mock_acompletion):
        """Custom rapporteur instructions are appended to the rapporteur call."""
        mock_acompletion.return_value = self.create_mock_completion('{"response": "yes", "confidence_score": 0.9}')
        opts = {'rapporteur_prompt': 'CUSTOM_INSTRUCTION'}
        asyncio.run(self.magi.run(CAPITAL_PUNISHMENT, method="VoteYesNo", method_options=opts))
        found = any(
            "CUSTOM_INSTRUCTION" in call[1].get('messages', [{}])[-1].get('content', '')
            for call in mock_acompletion.call_args_list
        )
        self.assertTrue(found)

    # ------------------------------------------------------------------
    # Model error surfacing
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_model_errors_surfaced_in_report(self, mock_acompletion):
        """When one model fails, its error appears in the report under Model Status."""
        from litellm.exceptions import NotFoundError

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                return self.create_mock_completion('{"response": "yes", "reason": "Reason A", "confidence_score": 0.9}')
            raise NotFoundError(
                message="models/bad-model is not found",
                model="modelB",
                llm_provider="test",
            )

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))
        self.assertIn("Model Status", result)
        self.assertIn("modelB", result)
        self.assertIn("[not_found]", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_fallback_chain_used_on_not_found(self, mock_acompletion):
        """When the primary model returns NotFoundError, the fallback is used."""
        from litellm.exceptions import NotFoundError

        config = {'llms': [['modelA', 'modelB']], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                raise NotFoundError(message="modelA not found", model="modelA", llm_provider="test")
            return self.create_mock_completion('{"response": "yes", "reason": "Fallback answer", "confidence_score": 0.8}')

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            result = asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))

        self.assertIn("modelB", result)
        self.assertIn("[fallback]", result)
        self.assertIn("modelA", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_quorum_not_met_returns_error(self, mock_acompletion):
        """When fewer models respond than min_models, run() returns a quorum error."""
        from litellm.exceptions import NotFoundError

        config = {'llms': ['modelA', 'modelB'], 'defaults': {'max_retries': 0, 'min_models': 2}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            if model == 'modelA':
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            raise NotFoundError(message="modelB not found", model="modelB", llm_provider="test")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            result = asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))

        self.assertIn("Quorum not met", result)
        self.assertIn("minimum required: 2", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_check_models_returns_status(self, mock_acompletion):
        """check_models() returns one entry per slot with ok/category/message."""
        from litellm.exceptions import NotFoundError

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            if model == 'modelA':
                return self.create_mock_completion("OK")
            raise NotFoundError(message="modelB is deprecated", model="modelB", llm_provider="test")

        mock_acompletion.side_effect = side_effect
        results = asyncio.run(self.magi.check_models(['modelA', 'modelB']))

        self.assertEqual(len(results), 2)
        a = results[0]['checks'][0]
        b = results[1]['checks'][0]
        self.assertTrue(a['ok'])
        self.assertEqual(a['category'], 'ok')
        self.assertFalse(b['ok'])
        self.assertIn(b['category'], ('deprecated', 'not_found'))

    # ------------------------------------------------------------------
    # Compose deliberative
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_compose_deliberative(self, mock_acompletion):
        """Compose supports deliberative rounds."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Compose" in content:
                return self.create_mock_completion('{"response": "Content", "confidence_score": 1.0}')
            elif "Rate" in content:
                return self.create_mock_completion(json.dumps({"response": {"Candidate X": {"score": 5, "justification": "ok"}}, "confidence_score": 1.0}))
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(
            "Write a one-sentence argument for why lying is sometimes morally permissible.",
            method="Compose",
            deliberative=True
        ))
        self.assertIn("Pre-Deliberation", result)
        self.assertIn("Post-Deliberation", result)


    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def test_empty_prompt_returns_error(self):
        """run() with an empty prompt returns an error immediately."""
        result = asyncio.run(self.magi.run("", method="VoteYesNo"))
        self.assertIn("Error", result)
        self.assertIn("empty", result.lower())

    def test_whitespace_only_prompt_returns_error(self):
        """run() with a whitespace-only prompt returns an error."""
        result = asyncio.run(self.magi.run("   ", method="VoteYesNo"))
        self.assertIn("Error", result)

    def test_empty_llms_returns_error(self):
        """run() with no configured LLMs returns a clear error."""
        magi = Magi({'llms': [], 'defaults': {}}, MOCK_PROMPTS)
        result = asyncio.run(magi.run("Some question", method="VoteYesNo"))
        self.assertIn("Error", result)
        self.assertIn("LLM", result)

    # ------------------------------------------------------------------
    # Confidence score safety
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_non_numeric_confidence_defaults_to_zero(self, mock_acompletion):
        """Non-numeric confidence_score is silently replaced with 0.0."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": "very confident"}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))
        # Should complete without raising ValueError; confidence defaults to 0.0
        self.assertIn("Result:", result)

    # ------------------------------------------------------------------
    # Null response field
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_null_response_field_triggers_retry(self, mock_acompletion):
        """A null 'response' field causes a retry; succeeds on the second attempt."""
        call_counts = {"modelA": 0, "modelB": 0}

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                call_counts['modelA'] += 1
                if call_counts['modelA'] == 1:
                    return self.create_mock_completion('{"response": null, "reason": "R", "confidence_score": 0.8}')
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.8}')
            return self.create_mock_completion('{"response": "no", "reason": "R", "confidence_score": 0.7}')

        config = {'llms': ['modelA', 'modelB'], 'defaults': {'max_retries': 1, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)
        mock_acompletion.side_effect = side_effect
        result = asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))
        self.assertGreater(call_counts['modelA'], 1, "modelA should have been called more than once due to null response")
        self.assertIn("Result:", result)

    # ------------------------------------------------------------------
    # asyncio.gather exception safety
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_unexpected_exception_in_gather_is_handled(self, mock_acompletion):
        """An unexpected exception from one slot does not crash the whole deliberation."""
        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            raise RuntimeError("Unexpected internal error")

        mock_acompletion.side_effect = side_effect
        # With min_models=1, the run should still produce a report from modelA
        result = asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))
        self.assertIn("Result:", result)

    # ------------------------------------------------------------------
    # Timeout
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_timeout_passed_to_acompletion(self, mock_acompletion):
        """request_timeout from config is forwarded to every acompletion call."""
        config = {'llms': ['modelA', 'modelB'], 'defaults': {'max_retries': 0, 'min_models': 1, 'request_timeout': 30}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))

        for call in mock_acompletion.call_args_list:
            self.assertEqual(call[1].get('timeout'), 30, "timeout must be 30 in every acompletion call")

    def test_default_timeout_is_60(self):
        """When request_timeout is absent from config, it defaults to 60 seconds."""
        magi = Magi({'llms': ['modelA'], 'defaults': {}}, MOCK_PROMPTS)
        self.assertEqual(magi.request_timeout, 60.0)

    # ------------------------------------------------------------------
    # Rate-limit fallback
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_rate_limit_triggers_fallback(self, mock_acompletion):
        """A RateLimitError on the primary model causes the fallback to be tried."""
        from litellm.exceptions import RateLimitError

        config = {'llms': [['modelA', 'modelB']], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                raise RateLimitError(message="Rate limit hit", model="modelA", llm_provider="test")
            return self.create_mock_completion('{"response": "yes", "reason": "Fallback answer", "confidence_score": 0.8}')

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            result = asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))

        self.assertIn("modelB", result)
        self.assertIn("[fallback]", result)

    # ------------------------------------------------------------------
    # Config schema validation
    # ------------------------------------------------------------------
    def test_config_validation_llms_not_a_list(self):
        """llms must be a list; passing a string raises ValueError."""
        with self.assertRaises(ValueError, msg="Expected ValueError for non-list llms"):
            Magi({'llms': 'openai/gpt-4'}, MOCK_PROMPTS)

    def test_config_validation_empty_model_string(self):
        """An empty string in llms raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': ['']}, MOCK_PROMPTS)

    def test_config_validation_empty_fallback_list(self):
        """An empty fallback list inside llms raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [[]]}, MOCK_PROMPTS)

    def test_config_validation_invalid_slot_type(self):
        """A non-string, non-list slot in llms raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [42]}, MOCK_PROMPTS)

    def test_config_validation_defaults_not_a_dict(self):
        """defaults must be a dict; passing a list raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [], 'defaults': ['oops']}, MOCK_PROMPTS)

    def test_config_validation_negative_max_retries(self):
        """Negative max_retries raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [], 'defaults': {'max_retries': -1}}, MOCK_PROMPTS)

    def test_config_validation_zero_timeout(self):
        """request_timeout of 0 raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [], 'defaults': {'request_timeout': 0}}, MOCK_PROMPTS)

    def test_config_validation_negative_timeout(self):
        """Negative request_timeout raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [], 'defaults': {'request_timeout': -5}}, MOCK_PROMPTS)

    def test_config_validation_invalid_vote_threshold(self):
        """vote_threshold outside (0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            Magi({'llms': [], 'defaults': {'vote_threshold': 1.5}}, MOCK_PROMPTS)

    def test_config_validation_valid_config_passes(self):
        """A well-formed config with all optional keys should construct without error."""
        config = {
            'llms': ['modelA', ['modelB', 'modelC']],
            'defaults': {
                'max_retries': 2,
                'min_models': 1,
                'request_timeout': 45,
                'vote_threshold': 0.6,
            },
        }
        magi = Magi(config, MOCK_PROMPTS)
        self.assertEqual(magi.request_timeout, 45.0)

    # ------------------------------------------------------------------
    # CLI argument validation
    # ------------------------------------------------------------------
    def test_vote_threshold_validation(self):
        """build_parser() accepts valid thresholds; main() rejects out-of-range values."""
        from magi_core.cli import build_parser
        parser = build_parser()

        args = parser.parse_args(["q", "--vote-threshold", "0.6"])
        self.assertEqual(args.vote_threshold, 0.6)

        args_zero = parser.parse_args(["q", "--vote-threshold", "0.0"])
        self.assertEqual(args_zero.vote_threshold, 0.0)  # parser accepts it; main() rejects it


    # ------------------------------------------------------------------
    # run_structured — JSON output
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_returns_dict(self, mock_acompletion):
        """run_structured() returns a dict, not a string."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        self.assertIsInstance(result, dict)

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_schema_fields(self, mock_acompletion):
        """run_structured() result contains all required top-level schema fields."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        for key in ("schema_version", "method", "prompt", "deliberative", "models", "rounds"):
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["method"], "VoteYesNo")
        self.assertFalse(result["deliberative"])

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_vote_aggregate(self, mock_acompletion):
        """run_structured() VoteYesNo result contains votes and winner in aggregate."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        round1 = result["rounds"][0]
        self.assertIn("aggregate", round1)
        self.assertIn("votes", round1["aggregate"])
        self.assertIn("winner", round1["aggregate"])
        self.assertIn("threshold", round1["aggregate"])
        self.assertEqual(round1["aggregate"]["winner"], "yes")

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_responses_have_pseudonyms(self, mock_acompletion):
        """Each response in run_structured() output has a pseudonym and model name."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        for resp in result["rounds"][0]["responses"]:
            self.assertIn("model", resp)
            self.assertIn("pseudonym", resp)
            self.assertTrue(resp["pseudonym"].startswith("Participant "))

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_deliberative_has_two_rounds(self, mock_acompletion):
        """run_structured() with deliberative=True returns two rounds."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content or "Deliberative" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo", deliberative=True))
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["rounds"][0]["round"], 1)
        self.assertEqual(result["rounds"][1]["round"], 2)

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_errors_in_round(self, mock_acompletion):
        """Model errors appear in round errors array in run_structured() output."""
        from litellm.exceptions import NotFoundError

        async def side_effect(*args, **kwargs):
            model = kwargs.get('model')
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            if model == 'modelA':
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            raise NotFoundError(message="modelB not found", model="modelB", llm_provider="test")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        errors = result["rounds"][0]["errors"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["model"], "modelB")
        self.assertIn(errors[0]["error_category"], ("not_found", "deprecated"))

    def test_run_structured_error_returns_error_key(self):
        """run_structured() with empty prompt returns a dict with an 'error' key."""
        result = asyncio.run(self.magi.run_structured("", method="VoteYesNo"))
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    @patch('magi_core.core.litellm.acompletion')
    def test_run_structured_is_json_serialisable(self, mock_acompletion):
        """run_structured() output can be serialised to JSON without errors."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run_structured(TROLLEY_PROBLEM, method="VoteYesNo"))
        try:
            serialised = json.dumps(result)
            self.assertIsInstance(serialised, str)
        except (TypeError, ValueError) as e:
            self.fail(f"run_structured() result is not JSON serialisable: {e}")


    # ------------------------------------------------------------------
    # Per-call API key passing
    # ------------------------------------------------------------------
    @patch('magi_core.core.litellm.acompletion')
    def test_api_key_forwarded_to_acompletion(self, mock_acompletion):
        """api_keys={'openai': 'sk-test'} is forwarded as api_key= to every acompletion call."""
        config = {'llms': ['openai/gpt-4o'], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo", api_keys={"openai": "sk-test"}))

        for c in mock_acompletion.call_args_list:
            self.assertEqual(c[1].get('api_key'), 'sk-test',
                             "Every acompletion call should have api_key='sk-test'")

    @patch('magi_core.core.litellm.acompletion')
    def test_api_key_not_sent_when_omitted(self, mock_acompletion):
        """Without api_keys, no api_key kwarg is passed to acompletion."""
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        asyncio.run(self.magi.run(TROLLEY_PROBLEM, method="VoteYesNo"))

        for c in mock_acompletion.call_args_list:
            self.assertNotIn('api_key', c[1],
                             "api_key should not appear in acompletion kwargs when omitted")

    @patch('magi_core.core.litellm.acompletion')
    def test_api_key_provider_mismatch_falls_back(self, mock_acompletion):
        """api_keys for a different provider does not inject api_key for the model's calls."""
        config = {'llms': ['openai/gpt-4o'], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo", api_keys={"anthropic": "sk-ant-xxx"}))

        for c in mock_acompletion.call_args_list:
            self.assertNotIn('api_key', c[1],
                             "api_key should not be passed when provider doesn't match")

    @patch('magi_core.core.litellm.acompletion')
    def test_api_key_multiple_providers(self, mock_acompletion):
        """Each model gets the correct provider-specific API key."""
        config = {'llms': ['openai/gpt-4o', 'anthropic/claude-3'], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)
        keys = {"openai": "sk-openai", "anthropic": "sk-anthropic"}

        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider'), \
             patch('magi_core.core.litellm.validate_environment', return_value={'keys_in_environment': True}):
            asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo", api_keys=keys))

        for c in mock_acompletion.call_args_list:
            model = c[1].get('model', '')
            if model.startswith('openai/'):
                self.assertEqual(c[1].get('api_key'), 'sk-openai')
            elif model.startswith('anthropic/'):
                self.assertEqual(c[1].get('api_key'), 'sk-anthropic')

    @patch('magi_core.core.litellm.acompletion')
    def test_check_models_with_api_keys(self, mock_acompletion):
        """check_models() forwards api_keys to _ping_model's acompletion call."""
        config = {'llms': ['openai/gpt-4o'], 'defaults': {}}
        magi = Magi(config, MOCK_PROMPTS)

        mock_acompletion.return_value = self.create_mock_completion("OK")

        asyncio.run(magi.check_models(api_keys={"openai": "sk-check"}))

        for c in mock_acompletion.call_args_list:
            self.assertEqual(c[1].get('api_key'), 'sk-check')

    @patch('magi_core.core.litellm.acompletion')
    def test_validate_model_skipped_when_api_key_provided(self, mock_acompletion):
        """When api_keys covers the model's provider, validate_environment is not called."""
        config = {'llms': ['openai/gpt-4o'], 'defaults': {'max_retries': 0, 'min_models': 1}}
        magi = Magi(config, MOCK_PROMPTS)

        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Vote" in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        with patch('magi_core.core.litellm.get_llm_provider') as mock_provider, \
             patch('magi_core.core.litellm.validate_environment') as mock_validate:
            asyncio.run(magi.run(TROLLEY_PROBLEM, method="VoteYesNo", api_keys={"openai": "sk-test"}))

        mock_validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
