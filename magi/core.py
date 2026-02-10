import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional
import litellm
from litellm.exceptions import NotFoundError
from magi.utils import generate_pseudonyms, resolve_tie

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Magi:
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self.config = config
        self.prompts = prompts
        self.llms = config.get('llms', [])
        self.max_retries = config.get('defaults', {}).get('max_retries', 3)
        
        if self.config.get('litellm_debug_mode', False):
            litellm._turn_on_debug()
        
    def _validate_model(self, model_name: str):
        """Validates that a model is available via litellm."""
        try:
            # Check provider/format
            litellm.get_llm_provider(model_name)
            
            # Check environment keys
            res = litellm.validate_environment(model_name)
            if not res.get('keys_in_environment'):
                missing = res.get('missing_keys', [])
                raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            raise ValueError(f"Model {model_name} is not available: {e}")

    def _build_rapporteur_prompt(self, method: str, user_prompt: str, responses_text: str, result_summary: Optional[str] = None) -> str:
        """Helper to build rapporteur prompt from template."""
        template = self.prompts.get('rapporteur_template', "")
        footer = self.prompts.get('rapporteur_footer', "")
        
        # Get settings for method
        settings = self.prompts.get('rapporteur', {}).get(method)
        if not settings:
            # Fallback for backward compatibility or error
             return f"Please summarize: {user_prompt}\n\n{responses_text}"

        context = settings.get('context', "").format(prompt=user_prompt)
        
        result_line = ""
        if settings.get('result_line') and result_summary:
             result_line = settings['result_line'].format(result=result_summary)
             
        response_type = settings.get('response_type', 'responses')
        instruction = settings.get('instruction', "")
        
        prompt = template.format(
            context=context,
            result_line=result_line,
            response_type=response_type,
            responses=responses_text,
            instruction=instruction,
            footer=footer
        )
        return prompt.strip()
        
    async def invoke_llm(self, model_name: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Invokes a single LLM and ensures JSON response.
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                # Prepare messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                if retries > 0:
                     # Add a hint to the model to correct its format
                     messages.append({"role": "user", "content": "Previous response was not valid JSON. Please ensure the response is a valid JSON object without markdown formatting."})

                # Call litellm (async)
                response = await litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"} 
                )
                
                content = response.choices[0].message.content
                
                # Parse JSON
                try:
                    data = json.loads(content)
                    # Normalize keys just in case
                    return {
                        "model": model_name,
                        "response": data.get("response"),
                        "reason": data.get("reason"),
                        "confidence_score": float(data.get("confidence_score", 0.0)),
                        "raw_response": content
                    }
                except json.JSONDecodeError:
                    # Basic cleanup try
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(clean_content)
                        return {
                            "model": model_name,
                            "response": data.get("response"),
                            "reason": data.get("reason"),
                            "confidence_score": float(data.get("confidence_score", 0.0)),
                            "raw_response": content
                        }
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON from {model_name} (Attempt {retries + 1}/{self.max_retries + 1}): {e}")
                        last_error = e
                        retries += 1
                        continue

            except NotFoundError as e:
                logger.error(f"Model {model_name} not found by provider: {e}")
                return {
                    "model": model_name,
                    "error": f"Model Not Found: {e.message if hasattr(e, 'message') else str(e)}",
                    "confidence_score": 0.0
                }

            except Exception as e:
                logger.warning(f"Error invoking {model_name} (Attempt {retries + 1}/{self.max_retries + 1}): {e}")
                last_error = e
                retries += 1
                
        # If we reach here, retries exhausted
        logger.error(f"Failed to get valid response from {model_name} after {self.max_retries} retries. Last error: {last_error}")
        return {
            "model": model_name,
            "error": f"Max retries exceeded. Last error: {last_error}",
            "confidence_score": 0.0
        }


    async def invoke_raw_llm(self, model_name: str, system_prompt: str, user_prompt: str) -> str:
        """Helper for text-only response (used by Rapporteur)"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = await litellm.acompletion(model=model_name, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error invoking raw {model_name}: {e}")
            return f"Error: {e}"

    async def run(self, user_prompt: str, system_prompt: Optional[str] = None, selected_llms: Optional[List[str]] = None, method: str = 'VoteYesNo', method_options: Dict[str, Any] = None) -> str:
        """
        Main entry point to run the aggregation.
        """
        if method_options is None:
            method_options = {}

        if not selected_llms:
            selected_llms = self.llms
        
        # Validate models
        for model in selected_llms:
            self._validate_model(model)

        # 1. Prepare Prompts
        sys_base = self.prompts['system_base']
        if system_prompt:
            sys_full = f"{system_prompt}\n\n{sys_base}"
        else:
            sys_full = sys_base

        method_config = self.prompts['methods'].get(method)
        if not method_config:
            raise ValueError(f"Unknown method: {method}")

        # specific prompt construction
        if method in ['VoteYesNo', 'VoteOptions']:
            options = method_options.get('options')
            if not options:
                if method == 'VoteYesNo':
                    options = method_config.get('default_options')
                else:
                    # If CLI didn't pass list, check config just in case, or fail
                    raise ValueError("VoteOptions method requires options to be specified (e.g. --options 'A,B,C').")
            
            # Make sure options is a list if it's a string
            if isinstance(options, str):
                options = [x.strip() for x in options.split(',')]
            
            # For formatting prompt, we might want string representation or list
            prompt_text = method_config['instruction'].format(options=options, prompt=user_prompt)
            if not method_options.get('allow_abstain', True):
                 prompt_text += " Abstaining is not allowed."
        elif method == 'Probability':
             allow_abstain = method_options.get('allow_abstain', True)
             if allow_abstain:
                 abstain_instr = "If you are truly unsure and cannot provide a probability, set `confidence_score` to -1.0."
             else:
                 abstain_instr = "You must provide a probability between 0.0 and 1.0. Do not use -1.0."
             
             prompt_text = method_config['instruction'].format(prompt=user_prompt, abstain_instruction=abstain_instr)
        else:
             prompt_text = method_config['instruction'].format(prompt=user_prompt)

        # 2. Invoke LLMs in parallel
        tasks = [self.invoke_llm(model, sys_full, prompt_text) for model in selected_llms]
        results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return "Error: No valid responses from LLMs."

        # 3. Aggregation Logic
        
        # Assign Pseudonyms
        mapping = generate_pseudonyms([r['model'] for r in valid_results])
        anonymized_responses = []
        for r in valid_results:
            pseudo = mapping[r['model']]
            anonymized_responses.append(f"{pseudo}: Response: {r['response']}, Reason: {r['reason']}, Confidence: {r['confidence_score']}")
        
        responses_text = "\n\n".join(anonymized_responses)

        final_output = ""
        rapporteur_name = None
        
        if method in ['VoteYesNo', 'VoteOptions']:
            # Count votes
            votes = {}
            # Count each vote
            for r in valid_results:
                vote = str(r['response']).lower()
                votes[vote] = votes.get(vote, 0) + 1
            
            total_votes = len(valid_results)
            threshold = method_options.get('vote_threshold', 0.5)
            
            # Determine winner
            winner = "No Majority"
            
            # If threshold is percentage of total
            for option, count in votes.items():
                if count / total_votes > threshold:
                     winner = option
                     # Break if found
                     break
            
            result_summary = f"Votes: {votes}. Winner: {winner} (Threshold: >{threshold:.0%})"
            
            # Select Rapporteur: Highest confidence who voted for the winner
            if winner != "No Majority":
                candidates = [r for r in valid_results if str(r['response']).lower() == winner]
            else:
                candidates = valid_results # Fallback
            
            if not candidates:
                 candidates = valid_results

            # Tie breaking
            if candidates:
                max_conf = max(c['confidence_score'] for c in candidates)
                best_candidates = [c for c in candidates if c['confidence_score'] == max_conf]
                rapporteur_data = resolve_tie(best_candidates)
                rapporteur_name = rapporteur_data['model']
            else:
                # Should not happen given logic above, but safe guard
                rapporteur_name = valid_results[0]['model']
            
            # Rapporteur Prompt
            rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text, result_summary)
            if method_options.get('rapporteur_prompt'):
                rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"

            summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
            
            # Vote Breakdown
            breakdown = []
            for r in valid_results:
                 breakdown.append(f" - {r['model']}: {r['response']} (Confidence: {r['confidence_score']:.2f})")
            breakdown_str = "\n".join(breakdown)
            
            final_output = f"Result: {winner}\nDetails: {result_summary}\n\nVote Breakdown:\n{breakdown_str}\n\nRapporteur ({rapporteur_name}) Summary:\n{summary}"

        elif method == 'Majority':
             # "Ask the LLM gives the highest confidence score to summarise the majoirty views"
             max_conf = max(r['confidence_score'] for r in valid_results)
             best_candidates = [r for r in valid_results if r['confidence_score'] == max_conf]
             rapporteur_data = resolve_tie(best_candidates)
             rapporteur_name = rapporteur_data['model']

             rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text)
             if method_options.get('rapporteur_prompt'):
                 rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"
             summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
             final_output = f"Majority View Summary (by {rapporteur_name}):\n{summary}"

        elif method == 'Consensus':
             # "Ask the LLM gives the highest confidence score to identify the common ground"
             max_conf = max(r['confidence_score'] for r in valid_results)
             best_candidates = [r for r in valid_results if r['confidence_score'] == max_conf]
             rapporteur_data = resolve_tie(best_candidates)
             rapporteur_name = rapporteur_data['model']

             rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text)
             if method_options.get('rapporteur_prompt'):
                 rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"
             summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
             final_output = f"Consensus/Common Ground (by {rapporteur_name}):\n{summary}"

        elif method == 'Minority':
             # "Ask the LLM gives the lowest confidence score to gaps"
             min_conf = min(r['confidence_score'] for r in valid_results)
             worst_candidates = [r for r in valid_results if r['confidence_score'] == min_conf]
             rapporteur_data = resolve_tie(worst_candidates)
             rapporteur_name = rapporteur_data['model']

             rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text)
             if method_options.get('rapporteur_prompt'):
                 rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"
             summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
             final_output = f"Minority/Gap Analysis (by {rapporteur_name}):\n{summary}"

        elif method == 'Probability':
             # Filter out abstentions (-1.0)
             filtered_results = [r for r in valid_results if r['confidence_score'] != -1.0]
             
             if not filtered_results:
                 if method_options.get('allow_abstain', True):
                      return "Result: All LLMs abstained."
                 else:
                      return "Error: No valid probabilities provided despite forced mode."

             # Calculate average and median
             conf_scores = [r['confidence_score'] for r in filtered_results]
             avg_conf = sum(conf_scores) / len(conf_scores)
             
             sorted_results = sorted(filtered_results, key=lambda x: x['confidence_score'])
             n = len(sorted_results)
             # Median selection
             median_idx = n // 2
             median_result = sorted_results[median_idx]
             median_conf = median_result['confidence_score']
             
             # In case of tie for median value, pick random (resolve_tie handles list)
             median_candidates = [r for r in filtered_results if r['confidence_score'] == median_conf]
             rapporteur_data = resolve_tie(median_candidates)
             rapporteur_name = rapporteur_data['model']
             
             min_conf = min(conf_scores)
             max_conf = max(conf_scores)
             
             result_summary = f"Average: {avg_conf:.2f} (Median: {median_conf:.2f}, Range: {min_conf:.2f} - {max_conf:.2f})"
             
             # Rebuild responses_text to exclude abstainers or include them? 
             # "discard this result" implies excluding.
             # Let's filter responses_text for consistency with result_summary
             filtered_responses = []
             for r in filtered_results:
                 pseudo = mapping[r['model']]
                 filtered_responses.append(f"{pseudo}: Response: {r['response']}, Reason: {r['reason']}, Confidence: {r['confidence_score']}")
             responses_text_prob = "\n\n".join(filtered_responses)
             
             rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text_prob, result_summary)
             if method_options.get('rapporteur_prompt'):
                 rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"
                 
             summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
             final_output = f"Probability Analysis:\n{result_summary}\n\nRapporteur ({rapporteur_name}) Summary:\n{summary}"

        # De-anonymize in final output
        for name, pseudo in mapping.items():
            final_output = final_output.replace(pseudo, name)

        return final_output
