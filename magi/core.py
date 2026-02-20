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

    def _prepare_base_prompt(self, method: str, user_prompt: str, method_options: Dict[str, Any]) -> str:
        """Helper to prepare the base prompt for a method."""
        method_config = self.prompts['methods'].get(method)
        if not method_config:
            raise ValueError(f"Unknown method: {method}")

        if method in ['VoteYesNo', 'VoteOptions']:
            options = method_options.get('options')
            if not options:
                if method == 'VoteYesNo':
                    options = method_config.get('default_options')
                else:
                    raise ValueError("VoteOptions method requires options to be specified (e.g. --options 'A,B,C').")
            
            if isinstance(options, str):
                options = [x.strip() for x in options.split(',')]
            
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
        
        return prompt_text

    async def _gather_responses(self, selected_llms: List[str], system_prompt: str, prompts: Any) -> List[Dict[str, Any]]:
        """
        Executes the round. prompts can be a single string (for all) or a dict {model: prompt}.
        """
        tasks = []
        for model in selected_llms:
            if isinstance(prompts, dict):
                p_text = prompts.get(model, prompts.get('default', ""))
            else:
                p_text = prompts
            
            tasks.append(self.invoke_llm(model, system_prompt, p_text))
            
        results = await asyncio.gather(*tasks)
        return [r for r in results if "error" not in r]

    async def _generate_report(self, method: str, user_prompt: str, method_options: Dict[str, Any], valid_results: List[Dict[str, Any]]) -> str:
        """Generates the report from results."""
        if not valid_results:
            return "Error: No valid responses from LLMs."

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
            for r in valid_results:
                vote = str(r['response']).lower()
                votes[vote] = votes.get(vote, 0) + 1
            
            total_votes = len(valid_results)
            threshold = method_options.get('vote_threshold', 0.5)
            
            winner = "No Majority"
            for option, count in votes.items():
                if count / total_votes > threshold:
                     winner = option
                     break
            
            result_summary = f"Votes: {votes}. Winner: {winner} (Threshold: >{threshold:.0%})"
            
            if winner != "No Majority":
                candidates = [r for r in valid_results if str(r['response']).lower() == winner]
            else:
                candidates = valid_results
            
            if not candidates:
                 candidates = valid_results

            if candidates:
                max_conf = max(c['confidence_score'] for c in candidates)
                best_candidates = [c for c in candidates if c['confidence_score'] == max_conf]
                rapporteur_data = resolve_tie(best_candidates)
                rapporteur_name = rapporteur_data['model']
            else:
                rapporteur_name = valid_results[0]['model']
            
            rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text, result_summary)
            if method_options.get('rapporteur_prompt'):
                rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"

            summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
            
            breakdown = []
            for r in valid_results:
                 breakdown.append(f" - {r['model']}: {r['response']} (Confidence: {r['confidence_score']:.2f})")
            breakdown_str = "\n".join(breakdown)
            
            final_output = f"Result: {winner}\nDetails: {result_summary}\n\nVote Breakdown:\n{breakdown_str}\n\nRapporteur ({rapporteur_name}) Summary:\n{summary}"

        elif method == 'Majority':
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
             min_conf = min(r['confidence_score'] for r in valid_results)
             worst_candidates = [r for r in valid_results if r['confidence_score'] == min_conf]
             rapporteur_data = resolve_tie(worst_candidates)
             rapporteur_name = rapporteur_data['model']

             rap_prompt = self._build_rapporteur_prompt(method, user_prompt, responses_text)
             if method_options.get('rapporteur_prompt'):
                 rap_prompt += f"\n\n{method_options.get('rapporteur_prompt')}"
             summary = await self.invoke_raw_llm(rapporteur_name, "You are a helpful assistant.", rap_prompt)
             final_output = f"Minority/Gap Analysis (by {rapporteur_name}):\n{summary}"

        elif method == 'Compose':
             # 1. Prepare candidates
             candidates_text = []
             candidate_map = {} # Pseudo -> Model
             model_to_pseudo = generate_pseudonyms([r['model'] for r in valid_results])
             
             for r in valid_results:
                 pseudo = model_to_pseudo[r['model']]
                 candidate_map[pseudo] = r['model']
                 candidates_text.append(f"--- Candidate {pseudo} ---\n{r['response']}\n-----------------------")
             
             all_candidates_str = "\n".join(candidates_text)
             
             # 2. Ask for ratings
             rating_prompt_template = self.prompts['methods']['Compose'].get('rating_instruction', "")
             if not rating_prompt_template:
                 return "Error: Missing rating instruction for Compose."
                 
             rating_prompt = rating_prompt_template.format(prompt=user_prompt, candidates=all_candidates_str)
             
             rating_tasks = []
             raters = [r['model'] for r in valid_results]
             
             for model in raters:
                 rating_tasks.append(self.invoke_llm(model, self.prompts['system_base'], rating_prompt))
                 
             rating_results = await asyncio.gather(*rating_tasks)
             
             # 3. Aggregate Ratings
             scores = {pseudo: [] for pseudo in candidate_map}
             justifications = {pseudo: [] for pseudo in candidate_map}
             
             for res in rating_results:
                 if "error" in res:
                     continue
                 
                 rater_model = res['model']
                 ratings = res.get('response') # Expecting dict
                 
                 if isinstance(ratings, dict):
                     for cand_key, rating_data in ratings.items():
                         target_pseudo = None
                         for p in candidate_map:
                             if p in cand_key:
                                 target_pseudo = p
                                 break
                         
                         if target_pseudo:
                             try:
                                 val = float(rating_data.get('score', 0))
                                 just = rating_data.get('justification', '')
                                 scores[target_pseudo].append(val)
                                 justifications[target_pseudo].append(f"{rater_model}: {val} - {just}")
                             except (ValueError, TypeError):
                                 pass
            
             # 4. Compute Final Scores & Rank
             ranked = []
             for pseudo, vals in scores.items():
                 avg = sum(vals) / len(vals) if vals else 0
                 real_name = candidate_map[pseudo]
                 ranked.append({
                     "pseudo": pseudo,
                     "model": real_name,
                     "average": avg,
                     "details": justifications[pseudo],
                     "text": next((r['response'] for r in valid_results if r['model'] == real_name), "")
                 })
             
             ranked.sort(key=lambda x: x['average'], reverse=True)
             
             # 5. Format Output
             output_lines = [f"Compose Results for: {user_prompt}", "="*40]
             
             for i, item in enumerate(ranked, 1):
                 output_lines.append(f"\nRank {i}: {item['model']} (as {item['pseudo']})")
                 output_lines.append(f"Average Score: {item['average']:.2f}")
                 output_lines.append("-" * 20)
                 output_lines.append(f"Composition:\n{item['text']}")
                 output_lines.append("-" * 20)
                 output_lines.append("Peer Reviews:")
                 if item['details']:
                     for d in item['details']:
                         output_lines.append(f"  * {d}")
                 else:
                     output_lines.append("  (No ratings received)")
                 output_lines.append("="*40)
                 
             final_output = "\n".join(output_lines)

        elif method == 'Probability':
             filtered_results = [r for r in valid_results if r['confidence_score'] != -1.0]
             
             if not filtered_results:
                 if method_options.get('allow_abstain', True):
                      return "Result: All LLMs abstained."
                 else:
                      return "Error: No valid probabilities provided despite forced mode."

             conf_scores = [r['confidence_score'] for r in filtered_results]
             avg_conf = sum(conf_scores) / len(conf_scores)
             
             sorted_results = sorted(filtered_results, key=lambda x: x['confidence_score'])
             n = len(sorted_results)
             median_idx = n // 2
             median_result = sorted_results[median_idx]
             median_conf = median_result['confidence_score']
             
             median_candidates = [r for r in filtered_results if r['confidence_score'] == median_conf]
             rapporteur_data = resolve_tie(median_candidates)
             rapporteur_name = rapporteur_data['model']
             
             min_conf = min(conf_scores)
             max_conf = max(conf_scores)
             
             result_summary = f"Average: {avg_conf:.2f} (Median: {median_conf:.2f}, Range: {min_conf:.2f} - {max_conf:.2f})"
             
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

    async def run(self, user_prompt: str, system_prompt: Optional[str] = None, selected_llms: Optional[List[str]] = None, method: str = 'VoteYesNo', method_options: Dict[str, Any] = None, deliberative: bool = False) -> str:
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

        # 1. Prepare System Prompt
        sys_base = self.prompts['system_base']
        if system_prompt:
            sys_full = f"{system_prompt}\n\n{sys_base}"
        else:
            sys_full = sys_base

        # 2. Round 1
        base_prompt = self._prepare_base_prompt(method, user_prompt, method_options)
        
        # Invoke LLMs
        results1 = await self._gather_responses(selected_llms, sys_full, base_prompt)
        
        if not results1:
             return "Error: No valid responses from LLMs in Round 1."
             
        report1 = await self._generate_report(method, user_prompt, method_options, results1)
        
        if not deliberative:
            return report1
            
        # 3. Round 2 (Deliberative)
        mapping = generate_pseudonyms([r['model'] for r in results1])
        
        prompts_round2 = {}
        for model in selected_llms:
            other_responses = []
            for r in results1:
                if r['model'] != model:
                    pseudo = mapping[r['model']]
                    other_responses.append(f"Participant {pseudo}: {r['response']} (Reason: {r['reason']})")
            
            if not other_responses:
                prompts_round2[model] = base_prompt
            else:
                peer_text = "\n".join(other_responses)
                delib_instr = self.prompts.get('deliberative_instruction', "").format(
                    peer_responses=peer_text,
                    original_prompt=base_prompt
                )
                prompts_round2[model] = delib_instr

        results2 = await self._gather_responses(selected_llms, sys_full, prompts_round2)
        
        if not results2:
             return f"Pre-Deliberation Results:\n{report1}\n\n{'='*40}\n\nPost-Deliberation Results:\nError: No valid responses in Round 2."
             
        report2 = await self._generate_report(method, user_prompt, method_options, results2)
        
        return f"Pre-Deliberation Results:\n{report1}\n\n{'='*40}\n\nPost-Deliberation Results:\n{report2}"
