import unittest
import asyncio
from unittest.mock import patch, MagicMock
from magi.core import Magi

class TestMagi(unittest.TestCase):
    def setUp(self):
        self.config = {'llms': ['openai/gpt-4o', 'gemini/gemini-pro']}
        self.prompts = {
            'system_base': 'Base sys prompt',
            'methods': {
                'VoteYesNo': {
                    'instruction': 'Vote {options} {prompt}',
                    'default_options': ['yes', 'no', 'abstain']
                }
            },
            'rapporteur_template': 'Context {context} Result {result_line} Responses {responses} Instruction {instruction}',
            'rapporteur_footer': 'Footer',
            'rapporteur': {
                'VoteYesNo': {
                    'context': 'Ctx',
                    'result_line': 'Res {result}',
                    'instruction': 'Instr'
                }
            }
        }
        self.magi = Magi(self.config, self.prompts)

    @patch('magi.core.litellm.acompletion')
    def test_vote_flow(self, mock_completion):
        # Mock responses
        # Model 1
        mock_resp1 = MagicMock()
        mock_resp1.choices[0].message.content = '{"response": "yes", "reason": "good", "confidence_score": 0.9}'
        
        # Model 2
        mock_resp2 = MagicMock()
        mock_resp2.choices[0].message.content = '{"response": "no", "reason": "bad", "confidence_score": 0.8}'
        
        # Rapporteur response
        mock_resp3 = MagicMock()
        mock_resp3.choices[0].message.content = 'Summary text'

        # We need to handle multiple calls to acompletion
        # 1. model1 (vote)
        # 2. model2 (vote)
        # 3. rapporteur (summary)
        
        # side_effect can be a list or iterator
        mock_completion.side_effect = [mock_resp1, mock_resp2, mock_resp3]

        result = asyncio.run(self.magi.run("Test prompt", method='VoteYesNo'))
        
        self.assertIn("Result: No Majority", result) # 1 yes 1 no = 50% vs threshold > 50%? 1/2 is not > 0.5. It is == 0.5.
        self.assertIn("Summary text", result)

if __name__ == '__main__':
    unittest.main()
