import unittest
import asyncio
import json
from unittest.mock import patch, MagicMock
from magi.core import Magi

# Mock Prompt Configuration matching prompts.yaml structure
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
        }
    }
}

class TestMagi(unittest.TestCase):
    def setUp(self):
        self.config = {'llms': ['modelA', 'modelB'], 'defaults': {'max_retries': 0}}
        self.magi = Magi(self.config, MOCK_PROMPTS)
        
        # Patch validation functions globally for this test class
        self.patcher1 = patch('magi.core.litellm.get_llm_provider')
        self.patcher2 = patch('magi.core.litellm.validate_environment')
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

    @patch('magi.core.litellm.acompletion')
    def test_vote_yes_no(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            user_msg = messages[-1]['content']
            
            if "Rapporteur" in user_msg:
                # Rapporteur phase
                return self.create_mock_completion("Summary of votes")
            elif "Vote" in user_msg:
                # Voting phase
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "yes", "reason": "A", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "no", "reason": "B", "confidence_score": 0.8}')
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        
        result = asyncio.run(self.magi.run("Should we do X?", method="VoteYesNo"))
        
        self.assertIn("Result: No Majority", result)
        self.assertIn("Summary of votes", result)

    @patch('magi.core.litellm.acompletion')
    def test_vote_yes_no_threshold(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "Vote" in messages[-1]['content']:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "yes", "reason": "A", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "no", "reason": "B", "confidence_score": 0.8}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        
        opts = {'vote_threshold': 0.4}
        result = asyncio.run(self.magi.run("Q", method="VoteYesNo", method_options=opts))
        
        # With threshold 0.4, 1 out of 2 (0.5) is > 0.4, so there is a winner.
        self.assertTrue("Result: yes" in result or "Result: no" in result)

    @patch('magi.core.litellm.acompletion')
    def test_vote_options(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "VoteOptions" in messages[-1]['content']:
                return self.create_mock_completion('{"response": "Option1", "reason": "Best", "confidence_score": 0.9}')
            return self.create_mock_completion("Summary")

        mock_acompletion.side_effect = side_effect
        
        opts = {'options': ['Option1', 'Option2']}
        result = asyncio.run(self.magi.run("Pick one", method="VoteOptions", method_options=opts))
        
        self.assertIn("result: option1", result.lower())

    @patch('magi.core.litellm.acompletion')
    def test_majority(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Majority" in content:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "Idea A", "reason": "A", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "Idea B", "reason": "B", "confidence_score": 0.7}')
            if kwargs.get('model') == 'modelA':
                return self.create_mock_completion("Majority Summary by A")
            return self.create_mock_completion("Wrong Summary")

        mock_acompletion.side_effect = side_effect
        
        result = asyncio.run(self.magi.run("Q", method="Majority"))
        self.assertIn("Majority Summary by A", result)

    @patch('magi.core.litellm.acompletion')
    def test_consensus(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            return self.create_mock_completion('{"response": "Common", "reason": "C", "confidence_score": 0.8}')

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run("Q", method="Consensus"))
        self.assertIn("Common", str(result) or "") # Check result somehow (the mocked response isn't directly in output unless summarized)
        # Actually Consensus output format is "Consensus/Common Ground (by ...): Summary"
        # Since invoke_raw_llm returns the mocked JSON string here (as I used same side_effect), let's fix that.
        
        async def smart_side_effect(*args, **kwargs):
             messages = kwargs.get('messages', [])
             if "Consensus" in messages[-1]['content']:
                 return self.create_mock_completion('{"response": "Common", "reason": "C", "confidence_score": 0.8}')
             else:
                 return self.create_mock_completion("Consensus Summary")
        
        mock_acompletion.side_effect = smart_side_effect
        result = asyncio.run(self.magi.run("Q", method="Consensus"))
        self.assertIn("Consensus Summary", result)

    @patch('magi.core.litellm.acompletion')
    def test_minority(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            if "Minority" in content:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "A", "reason": "A", "confidence_score": 0.9}')
                else:
                    return self.create_mock_completion('{"response": "B", "reason": "B", "confidence_score": 0.5}')
            
            if kwargs.get('model') == 'modelB':
                return self.create_mock_completion("Minority Report by B")
            return self.create_mock_completion("Wrong Report")

        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run("Q", method="Minority"))
        self.assertIn("Minority Report by B", result)

    @patch('magi.core.litellm.acompletion')
    def test_probability(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            if "Probability" in messages[-1]['content']:
                if kwargs.get('model') == 'modelA':
                    return self.create_mock_completion('{"response": "Likely", "reason": "A", "confidence_score": 0.8}')
                else:
                    return self.create_mock_completion('{"response": "Unsure", "reason": "B", "confidence_score": -1.0}')
            return self.create_mock_completion("Prob Summary")

        mock_acompletion.side_effect = side_effect
        
        result = asyncio.run(self.magi.run("Q", method="Probability"))
        self.assertIn("Average: 0.80", result)
        self.assertIn("Prob Summary", result)

    @patch('magi.core.litellm.acompletion')
    def test_compose(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            
            if "Compose" in content:
                return self.create_mock_completion('{"response": "Poem content", "reason": "Creative", "confidence_score": 1.0}')
            elif "Rate" in content:
                import re
                candidates = re.findall(r"--- Candidate (.*?) ---", content)
                ratings = {}
                for c in candidates:
                    ratings[c.strip()] = {"score": 8.0, "justification": "Nice"}
                
                return self.create_mock_completion(json.dumps({
                    "response": ratings,
                    "reason": "Rated",
                    "confidence_score": 1.0
                }))
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        
        result = asyncio.run(self.magi.run("Write a poem", method="Compose"))
        
        self.assertIn("Compose Results", result)
        self.assertIn("Rank 1:", result)
        self.assertIn("Poem content", result)
        self.assertIn("Average Score: 8.00", result)

    @patch('magi.core.litellm.acompletion')
    def test_deliberative_mode(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            messages = kwargs.get('messages', [])
            content = messages[-1]['content']
            
            if "Vote" in content and "Deliberative Instruction" not in content:
                return self.create_mock_completion('{"response": "yes", "reason": "R1", "confidence_score": 0.9}')
            elif "Rapporteur" in content:
                return self.create_mock_completion("Summary")
            elif "Deliberative Instruction" in content:
                return self.create_mock_completion('{"response": "no", "reason": "Changed mind", "confidence_score": 0.95}')
            
            return self.create_mock_completion('{}')

        mock_acompletion.side_effect = side_effect
        
        result = asyncio.run(self.magi.run("Q", method="VoteYesNo", deliberative=True))
        
        self.assertIn("Pre-Deliberation Results", result)
        self.assertIn("Post-Deliberation Results", result)

    @patch('magi.core.litellm.acompletion')
    def test_no_abstain(self, mock_acompletion):
        mock_acompletion.return_value = self.create_mock_completion('{"response": "yes", "confidence_score": 0.9}')
        
        opts = {'allow_abstain': False}
        asyncio.run(self.magi.run("Q", method="VoteYesNo", method_options=opts))
        
        found = False
        for call in mock_acompletion.call_args_list:
             messages = call[1].get('messages', [])
             if messages and "Abstaining is not allowed" in messages[-1]['content']:
                 found = True
                 break
        self.assertTrue(found)

    @patch('magi.core.litellm.acompletion')
    def test_custom_rapporteur_prompt(self, mock_acompletion):
        mock_acompletion.return_value = self.create_mock_completion('{"response": "yes", "confidence_score": 0.9}')
        
        opts = {'rapporteur_prompt': 'CUSTOM_INSTRUCTION'}
        asyncio.run(self.magi.run("Q", method="VoteYesNo", method_options=opts))
        
        found = False
        for call in mock_acompletion.call_args_list:
             messages = call[1].get('messages', [])
             if messages and "CUSTOM_INSTRUCTION" in messages[-1]['content']:
                 found = True
                 break
        self.assertTrue(found)

    @patch('magi.core.litellm.acompletion')
    def test_compose_deliberative(self, mock_acompletion):
        async def side_effect(*args, **kwargs):
            content = kwargs.get('messages', [{}])[-1].get('content', '')
            if "Compose" in content:
                return self.create_mock_completion('{"response": "Content", "confidence_score": 1.0}')
            elif "Rate" in content:
                return self.create_mock_completion(json.dumps({"response": {"Candidate X": {"score": 5, "justification": "ok"}}, "confidence_score": 1.0}))
            return self.create_mock_completion('{}')
        
        mock_acompletion.side_effect = side_effect
        result = asyncio.run(self.magi.run("Q", method="Compose", deliberative=True))
        self.assertIn("Pre-Deliberation", result)
        self.assertIn("Post-Deliberation", result)

if __name__ == '__main__':
    unittest.main()
