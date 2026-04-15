import asyncio
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import litellm
from litellm.exceptions import NotFoundError, AuthenticationError
from magi_core.utils import generate_pseudonyms, resolve_tie

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A slot is either a single model name or a priority-ordered fallback list.
Slot = Union[str, List[str]]


def _primary(slot: Slot) -> str:
    """Return the primary model name for a slot."""
    return slot if isinstance(slot, str) else slot[0]


def _reveal_names_in_text(text: str, mapping: Dict[str, str]) -> str:
    """Replace pseudonyms with their real model names in user-facing text.

    `mapping` maps real_model_name -> pseudonym (e.g. "openai/gpt-5.4-nano"
    -> "Participant X7K2"). Two passes per (model, pseudonym) pair:
      1. Absorb any preceding "Participant[s] " prefix (case-insensitive) so
         "Participant X7K2" / "participants X7K2" render cleanly.
      2. Replace bare 4-char IDs appearing as standalone alphanumeric runs so
         leaks like "JSVB's position" or "5PPK and 4TEY" are also rewritten.

    IDs are drawn from a restricted alphabet (see utils._ID_CHARS) that
    excludes I, L, O, 0, 1, so collisions with common user-written acronyms
    are rare; residual risk ≈ 1/32^4 per assignment.
    """
    if not text or not mapping:
        return text
    for real_name, pseudo in mapping.items():
        parts = pseudo.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        pid = parts[1]
        prefix_re = re.compile(
            r"[Pp]articipants?\s+" + re.escape(pid) + r"(?![A-Za-z0-9])"
        )
        text = prefix_re.sub(real_name, text)
        bare_re = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(pid) + r"(?![A-Za-z0-9])"
        )
        text = bare_re.sub(real_name, text)
    return text


def _match_pseudonym(cand_key: str, pseudonyms: List[str]) -> Optional[str]:
    """Resolve a reviewer-returned JSON key to one of the known pseudonyms.

    Matches case-insensitively on the full pseudonym first, then on the bare ID
    token (e.g. "X7K2") as a standalone alphanumeric run — so keys like
    "Candidate_X7K2" or "participant x7k2" still resolve.
    """
    if not isinstance(cand_key, str) or not cand_key:
        return None
    low = cand_key.lower()
    for p in pseudonyms:
        if p.lower() in low:
            return p
    for p in pseudonyms:
        parts = p.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        pid = parts[1]
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(pid) + r"(?![A-Za-z0-9])",
                     cand_key, re.IGNORECASE):
            return p
    return None


def _pair_ratings_with_pseudonyms(
    ratings: Any, pseudonyms_in_order: List[str]
) -> Tuple[List[Tuple[Optional[str], Any, str]], bool]:
    """Pair each rating entry with a pseudonym.

    Returns (pairs, positional) where pairs is a list of
    (pseudonym_or_None, rating_data, raw_key) and `positional` is True when
    attribution fell back to candidate ordering.

    Resolution order:
      * list response: attribute by index against `pseudonyms_in_order`.
      * dict response: match each key via `_match_pseudonym`; if *every* key
        fails to match and the arity equals len(pseudonyms_in_order), fall
        back to insertion-order attribution and flag as positional.
    """
    if isinstance(ratings, list):
        pairs: List[Tuple[Optional[str], Any, str]] = []
        for i, rd in enumerate(ratings):
            pseudo = pseudonyms_in_order[i] if i < len(pseudonyms_in_order) else None
            pairs.append((pseudo, rd, f"[{i}]"))
        return pairs, True
    if not isinstance(ratings, dict):
        return [], False
    items = list(ratings.items())
    resolved: List[Tuple[Optional[str], Any, str]] = [
        (_match_pseudonym(k, pseudonyms_in_order), v, str(k)) for k, v in items
    ]
    if (items
            and len(items) == len(pseudonyms_in_order)
            and all(r[0] is None for r in resolved)):
        return (
            [(pseudonyms_in_order[i], v, str(k)) for i, (k, v) in enumerate(items)],
            True,
        )
    return resolved, False


def _classify_error(exc: Exception) -> Tuple[str, str]:
    """
    Return (category, short_message) for an LLM exception.

    Categories:
      deprecated  — model has been removed by the provider (triggers fallback)
      not_found   — model name is unrecognised (triggers fallback)
      auth        — invalid or missing API key (triggers fallback)
      rate_limit  — quota exceeded (triggers fallback)
      timeout     — request timed out (retries primary only)
      unknown     — anything else (retries primary only)
    """
    try:
        from litellm.exceptions import RateLimitError, Timeout
    except ImportError:
        RateLimitError = Timeout = None

    msg = str(exc)

    if isinstance(exc, NotFoundError):
        lower = msg.lower()
        if any(kw in lower for kw in ("no longer available", "deprecated", "removed", "not supported")):
            return "deprecated", _extract_provider_message(msg)
        return "not_found", _extract_provider_message(msg)

    if isinstance(exc, AuthenticationError):
        return "auth", "Invalid or missing API key"

    if RateLimitError and isinstance(exc, RateLimitError):
        return "rate_limit", "Rate limit exceeded"

    if Timeout and isinstance(exc, Timeout):
        return "timeout", "Request timed out"

    return "unknown", msg[:120]


def _extract_provider_message(raw: str) -> str:
    """Pull the provider's human-readable message from a litellm exception string."""
    m = re.search(r'"message"\s*:\s*"([^"]+)"', raw)
    if m:
        return m.group(1)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1][:120] if lines else raw[:120]


class Magi:
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, Any]):
        self._validate_config(config)
        self.config = config
        self.prompts = prompts
        self.llms: List[Slot] = config.get('llms', [])
        defaults = config.get('defaults', {})
        self.max_retries = defaults.get('max_retries', 3)
        self.request_timeout: float = float(defaults.get('request_timeout', 60))

        self._api_keys: Optional[Dict[str, str]] = None

        if self.config.get('litellm_debug_mode', False):
            litellm._turn_on_debug()

    def _resolve_api_key(self, model_name: str) -> Optional[str]:
        """Return the per-call API key for a model, or None to use env vars."""
        if not self._api_keys:
            return None
        provider = model_name.split("/")[0] if "/" in model_name else None
        return self._api_keys.get(provider) if provider else None

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate config structure at construction time, raising ValueError on errors."""
        if not isinstance(config, dict):
            raise ValueError("config must be a dict")

        llms = config.get('llms')
        if llms is not None:
            if not isinstance(llms, list):
                raise ValueError("config['llms'] must be a list")
            for i, slot in enumerate(llms):
                if isinstance(slot, str):
                    if not slot.strip():
                        raise ValueError(f"config['llms'][{i}] must not be an empty string")
                elif isinstance(slot, list):
                    if not slot:
                        raise ValueError(f"config['llms'][{i}] fallback list must not be empty")
                    for j, model in enumerate(slot):
                        if not isinstance(model, str) or not model.strip():
                            raise ValueError(
                                f"config['llms'][{i}][{j}] must be a non-empty string, got {model!r}"
                            )
                else:
                    raise ValueError(
                        f"config['llms'][{i}] must be a string or list, got {type(slot).__name__}"
                    )

        defaults = config.get('defaults')
        if defaults is not None:
            if not isinstance(defaults, dict):
                raise ValueError("config['defaults'] must be a dict")
            for key in ('max_retries', 'min_models'):
                val = defaults.get(key)
                if val is not None and (not isinstance(val, int) or val < 0):
                    raise ValueError(
                        f"config['defaults']['{key}'] must be a non-negative integer, got {val!r}"
                    )
            timeout_val = defaults.get('request_timeout')
            if timeout_val is not None:
                try:
                    if float(timeout_val) <= 0:
                        raise ValueError()
                except (TypeError, ValueError):
                    raise ValueError(
                        f"config['defaults']['request_timeout'] must be a positive number, got {timeout_val!r}"
                    )
            threshold_val = defaults.get('vote_threshold')
            if threshold_val is not None:
                try:
                    t = float(threshold_val)
                    if not (0 < t <= 1.0):
                        raise ValueError()
                except (TypeError, ValueError):
                    raise ValueError(
                        f"config['defaults']['vote_threshold'] must be a float in (0, 1], got {threshold_val!r}"
                    )

    def _validate_model(self, model_name: str):
        """Validates that a model is configured and its API key is present."""
        try:
            litellm.get_llm_provider(model_name)
            # Skip env-var check when a per-call API key covers this model's provider
            if self._resolve_api_key(model_name):
                return
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

        settings = self.prompts.get('rapporteur', {}).get(method)
        if not settings:
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
        """Invokes a single LLM and ensures a structured JSON response."""
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                if retries > 0:
                    messages.append({"role": "user", "content": "Previous response was not valid JSON. Please ensure the response is a valid JSON object without markdown formatting."})

                extra_kwargs = {}
                api_key = self._resolve_api_key(model_name)
                if api_key:
                    extra_kwargs["api_key"] = api_key
                response = await litellm.acompletion(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=self.request_timeout,
                    **extra_kwargs,
                )

                content = response.choices[0].message.content

                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    clean = content.replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(clean)
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON from {model_name} (attempt {retries + 1}): {e}")
                        last_error = e
                        retries += 1
                        continue

                response_val = data.get("response")
                if response_val is None:
                    logger.warning(f"LLM {model_name} returned null 'response' field (attempt {retries + 1})")
                    last_error = ValueError("Response field is null")
                    retries += 1
                    continue

                try:
                    confidence = float(data.get("confidence_score", 0.0))
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric confidence_score from {model_name}, defaulting to 0.0")
                    confidence = 0.0

                return {
                    "model": model_name,
                    "response": response_val,
                    "reason": data.get("reason", ""),
                    "confidence_score": confidence,
                    "raw_response": content
                }

            except (NotFoundError, AuthenticationError) as e:
                # Permanent errors — do not retry
                category, short_msg = _classify_error(e)
                logger.error(f"Permanent error for {model_name} [{category}]: {short_msg}")
                return {"model": model_name, "error": short_msg, "error_category": category, "confidence_score": 0.0}

            except Exception as e:
                # Transient errors — retry
                logger.warning(f"Error invoking {model_name} (attempt {retries + 1}/{self.max_retries + 1}): {e}")
                last_error = e
                retries += 1

        category, short_msg = _classify_error(last_error) if last_error else ("unknown", "No response after retries")
        logger.error(f"Failed to get valid response from {model_name}: {short_msg}")
        return {"model": model_name, "error": short_msg, "error_category": category, "confidence_score": 0.0}

    async def invoke_raw_llm(self, model_name: str, system_prompt: str, user_prompt: str) -> str:
        """Text-only LLM call used by the rapporteur."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            extra_kwargs = {}
            api_key = self._resolve_api_key(model_name)
            if api_key:
                extra_kwargs["api_key"] = api_key
            response = await litellm.acompletion(model=model_name, messages=messages, timeout=self.request_timeout, **extra_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            _, short_msg = _classify_error(e)
            logger.error(f"Rapporteur error for {model_name}: {short_msg}")
            return f"[Rapporteur error: {short_msg}]"

    async def _ping_model(self, model_name: str) -> Tuple[bool, str, str]:
        """Make a minimal API call to verify a model is reachable and returns content."""
        try:
            extra_kwargs = {}
            api_key = self._resolve_api_key(model_name)
            if api_key:
                extra_kwargs["api_key"] = api_key
            r = await litellm.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with the word OK and nothing else."}],
                max_tokens=500,  # Generous limit — thinking models reserve tokens internally
                timeout=self.request_timeout,
                **extra_kwargs,
            )
            content = r.choices[0].message.content
            if content:
                return True, "ok", content.strip()[:30]
            return False, "empty_response", "Model returned empty content"
        except Exception as e:
            category, short_msg = _classify_error(e)
            return False, category, short_msg

    async def check_models(self, models: Optional[List[Slot]] = None, api_keys: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Probe each configured slot with a real API call and return availability results.
        Each entry in the returned list covers one slot (primary + any fallbacks).
        """
        self._api_keys = api_keys
        if models is None:
            models = self.llms

        results = []
        for slot in models:
            slot_models = [slot] if isinstance(slot, str) else list(slot)
            checks = []
            for model in slot_models:
                ok, category, message = await self._ping_model(model)
                checks.append({"model": model, "ok": ok, "category": category, "message": message})
            results.append({
                "primary": slot_models[0],
                "fallbacks": slot_models[1:],
                "checks": checks,
            })
        return results

    def _prepare_base_prompt(self, method: str, user_prompt: str, method_options: Dict[str, Any]) -> str:
        """Prepare the base prompt for a given method."""
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

    async def _invoke_slot(self, slot: Slot, system_prompt: str, prompt: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Invoke a slot (single model or fallback chain).

        Returns (result, None) on success, or (None, error_dict) if every option fails.
        Fallbacks are tried for permanent errors (deprecated, not_found, auth) and rate limits.
        Transient errors (timeout, unknown) exhaust retries on the primary only.
        """
        models = [slot] if isinstance(slot, str) else list(slot)
        last_error_result = None

        for i, model in enumerate(models):
            result = await self.invoke_llm(model, system_prompt, prompt)
            if "error" not in result:
                if i > 0:
                    # Record which primary this model substituted for
                    result["fallback_for"] = models[0]
                return result, None

            category = result.get("error_category", "unknown")
            last_error_result = result

            # Burn through the fallback chain for permanent errors and rate limits
            if category not in ("deprecated", "not_found", "auth", "rate_limit"):
                break

        if last_error_result and len(models) > 1:
            last_error_result["attempted_fallbacks"] = models

        return None, last_error_result

    async def _gather_responses(
        self,
        selected_llms: List[Slot],
        system_prompt: str,
        prompts: Any,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Execute one round of LLM calls across all slots (in parallel).

        prompts can be a single string shared by all, or a dict keyed by primary model name.
        Returns (valid_results, error_results).
        """
        tasks = []
        for slot in selected_llms:
            primary = _primary(slot)
            if isinstance(prompts, dict):
                p_text = prompts.get(primary, prompts.get('default', ""))
            else:
                p_text = prompts
            tasks.append(self._invoke_slot(slot, system_prompt, p_text))

        pairs = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        errors = []
        for item in pairs:
            if isinstance(item, BaseException):
                logger.error(f"Unexpected task exception in gather: {item}")
                errors.append({"model": "unknown", "error": str(item), "error_category": "unknown", "confidence_score": 0.0})
            else:
                r, e = item
                if r is not None:
                    valid.append(r)
                if e is not None:
                    errors.append(e)
        return valid, errors

    # ------------------------------------------------------------------
    # Structured result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _language_directive(language: Optional[str]) -> Optional[str]:
        """Return a brief system-prompt directive mandating the response language.

        Returns None when no language is given. Free text is accepted verbatim,
        so callers can specify variants like "British English" or "German".
        """
        if not language or not str(language).strip():
            return None
        return f"Respond in {str(language).strip()}."

    async def _build_round_data(
        self,
        round_num: int,
        method: str,
        user_prompt: str,
        method_options: Dict[str, Any],
        valid_results: List[Dict[str, Any]],
        error_results: List[Dict[str, Any]],
        language: Optional[str] = None,
        show_real_names_in_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Build structured data for one deliberation round, including the rapporteur call.
        Returns a dict that is both JSON-serialisable and used by the markdown renderer.
        """
        if not valid_results:
            return {
                "round": round_num,
                "responses": [],
                "errors": error_results,
                "aggregate": None,
                "rapporteur": None,
                "error": "No valid responses from LLMs.",
            }

        mapping = generate_pseudonyms([r["model"] for r in valid_results])

        responses = [
            {
                "model": r["model"],
                "pseudonym": mapping[r["model"]],
                "response": r["response"],
                "reason": r.get("reason", ""),
                "confidence_score": r["confidence_score"],
                "fallback_for": r.get("fallback_for"),
            }
            for r in valid_results
        ]

        anonymized_text = "\n\n".join(
            f"{mapping[r['model']]}: Response: {r['response']}, Reason: {r['reason']}, Confidence: {r['confidence_score']}"
            for r in valid_results
        )

        aggregate: Optional[Dict] = None
        rapporteur_model: Optional[str] = None
        rapporteur_summary: Optional[str] = None

        lang_directive = self._language_directive(language)
        rap_system_prompt = (
            f"{lang_directive} You are a helpful assistant." if lang_directive
            else "You are a helpful assistant."
        )

        if method in ("VoteYesNo", "VoteOptions"):
            votes: Dict[str, int] = {}
            for r in valid_results:
                vote = str(r["response"]).lower()
                votes[vote] = votes.get(vote, 0) + 1
            total = len(valid_results)
            threshold = method_options.get("vote_threshold", 0.5)
            winner = "No Majority"
            for option, count in votes.items():
                if count / total > threshold:
                    winner = option
                    break
            aggregate = {"votes": votes, "winner": winner, "threshold": threshold}

            candidates = (
                [r for r in valid_results if str(r["response"]).lower() == winner]
                if winner != "No Majority"
                else valid_results
            )
            if not candidates:
                candidates = valid_results
            max_conf = max(c["confidence_score"] for c in candidates)
            rapporteur_data = resolve_tie([c for c in candidates if c["confidence_score"] == max_conf])
            rapporteur_model = rapporteur_data["model"]

            result_summary = f"Votes: {votes}. Winner: {winner} (Threshold: >{threshold:.0%})"
            rap_prompt = self._build_rapporteur_prompt(method, user_prompt, anonymized_text, result_summary)
            if method_options.get("rapporteur_prompt"):
                rap_prompt += f"\n\n{method_options['rapporteur_prompt']}"
            rapporteur_summary = await self.invoke_raw_llm(rapporteur_model, rap_system_prompt, rap_prompt)

        elif method == "Probability":
            filtered = [r for r in valid_results if r["confidence_score"] != -1.0]
            if not filtered:
                agg: Dict[str, Any] = {"abstained": True}
                if not method_options.get("allow_abstain", True):
                    agg = {"error": "No valid probabilities provided despite forced mode."}
                return {
                    "round": round_num,
                    "responses": responses,
                    "errors": error_results,
                    "aggregate": agg,
                    "rapporteur": None,
                }
            scores = [r["confidence_score"] for r in filtered]
            avg = sum(scores) / len(scores)
            sorted_f = sorted(filtered, key=lambda x: x["confidence_score"])
            median_result = sorted_f[len(sorted_f) // 2]
            median_conf = median_result["confidence_score"]
            aggregate = {
                "average": round(avg, 4),
                "median": round(median_conf, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "abstained_count": len(valid_results) - len(filtered),
            }
            median_candidates = [r for r in filtered if r["confidence_score"] == median_conf]
            rapporteur_data = resolve_tie(median_candidates)
            rapporteur_model = rapporteur_data["model"]
            result_summary = f"Average: {avg:.2f} (Median: {median_conf:.2f}, Range: {min(scores):.2f} - {max(scores):.2f})"
            filtered_text = "\n\n".join(
                f"{mapping[r['model']]}: Response: {r['response']}, Reason: {r['reason']}, Confidence: {r['confidence_score']}"
                for r in filtered
            )
            rap_prompt = self._build_rapporteur_prompt(method, user_prompt, filtered_text, result_summary)
            if method_options.get("rapporteur_prompt"):
                rap_prompt += f"\n\n{method_options['rapporteur_prompt']}"
            rapporteur_summary = await self.invoke_raw_llm(rapporteur_model, rap_system_prompt, rap_prompt)

        elif method == "Compose":
            candidate_map = {mapping[r["model"]]: r["model"] for r in valid_results}
            all_candidates_str = "\n".join(
                f"--- Candidate {mapping[r['model']]} ---\n{r['response']}\n-----------------------"
                for r in valid_results
            )
            rating_prompt_template = self.prompts["methods"]["Compose"].get("rating_instruction", "")
            if not rating_prompt_template:
                return {
                    "round": round_num,
                    "responses": responses,
                    "errors": error_results,
                    "aggregate": {"error": "Missing rating instruction for Compose."},
                    "rapporteur": None,
                }
            rating_prompt = rating_prompt_template.format(prompt=user_prompt, candidates=all_candidates_str)
            rating_tasks = [
                self.invoke_llm(r["model"], self.prompts["system_base"], rating_prompt)
                for r in valid_results
            ]
            rating_results = await asyncio.gather(*rating_tasks, return_exceptions=True)

            scores_map: Dict[str, List[float]] = {pseudo: [] for pseudo in candidate_map}
            reviews_map: Dict[str, List[Dict]] = {pseudo: [] for pseudo in candidate_map}
            pseudonyms_in_order = [mapping[r["model"]] for r in valid_results]
            unattributed_reviewers: List[Dict[str, Any]] = []

            for res in rating_results:
                if isinstance(res, BaseException):
                    logger.error(f"Unexpected exception in Compose rating gather: {res}")
                    continue
                if "error" in res:
                    continue
                ratings = res.get("response")
                pairs, positional = _pair_ratings_with_pseudonyms(ratings, pseudonyms_in_order)
                reviewer_unattributed_keys: List[str] = []
                for pseudo, rating_data, raw_key in pairs:
                    if pseudo is None:
                        reviewer_unattributed_keys.append(raw_key)
                        continue
                    if not isinstance(rating_data, dict):
                        continue
                    try:
                        val = float(rating_data.get("score", 0))
                    except (ValueError, TypeError):
                        continue
                    just = rating_data.get("justification", "")
                    scores_map[pseudo].append(val)
                    review = {"reviewer_model": res["model"], "score": val, "justification": just}
                    if positional:
                        review["attribution"] = "positional"
                    reviews_map[pseudo].append(review)
                if reviewer_unattributed_keys:
                    logger.warning(
                        "Compose: reviewer %s returned unattributable keys: %s",
                        res.get("model"), reviewer_unattributed_keys,
                    )
                    unattributed_reviewers.append(
                        {"reviewer_model": res.get("model"), "raw_keys": reviewer_unattributed_keys}
                    )

            ranked = sorted(
                [
                    {
                        "rank": 0,
                        "model": candidate_map[pseudo],
                        "pseudonym": pseudo,
                        "average_score": round(sum(vals) / len(vals), 4) if vals else 0.0,
                        "text": next(
                            (r["response"] for r in valid_results if r["model"] == candidate_map[pseudo]), ""
                        ),
                        "peer_reviews": reviews_map[pseudo],
                    }
                    for pseudo, vals in scores_map.items()
                ],
                key=lambda x: x["average_score"],
                reverse=True,
            )
            for i, item in enumerate(ranked, 1):
                item["rank"] = i

            if show_real_names_in_report:
                for item in ranked:
                    for pr in item["peer_reviews"]:
                        pr["justification"] = _reveal_names_in_text(
                            pr.get("justification", ""), mapping
                        )

            aggregate_compose: Dict[str, Any] = {"ranked_candidates": ranked}
            if unattributed_reviewers:
                aggregate_compose["unattributed_reviewers"] = unattributed_reviewers
            return {
                "round": round_num,
                "responses": responses,
                "errors": error_results,
                "aggregate": aggregate_compose,
                "rapporteur": None,
            }

        else:
            # Majority, Consensus, Minority, Synthesis
            if method == "Minority":
                min_conf = min(r["confidence_score"] for r in valid_results)
                rapporteur_data = resolve_tie([r for r in valid_results if r["confidence_score"] == min_conf])
            else:
                max_conf = max(r["confidence_score"] for r in valid_results)
                rapporteur_data = resolve_tie([r for r in valid_results if r["confidence_score"] == max_conf])
            rapporteur_model = rapporteur_data["model"]
            rap_prompt = self._build_rapporteur_prompt(method, user_prompt, anonymized_text)
            if method_options.get("rapporteur_prompt"):
                rap_prompt += f"\n\n{method_options['rapporteur_prompt']}"
            rapporteur_summary = await self.invoke_raw_llm(rapporteur_model, rap_system_prompt, rap_prompt)

        # De-anonymise rapporteur summary (replace pseudonyms with real model names).
        # Gated by show_real_names_in_report; uses regex pipeline that catches
        # bare 4-char IDs as well as the full "Participant XXXX" form.
        if rapporteur_summary and show_real_names_in_report:
            rapporteur_summary = _reveal_names_in_text(rapporteur_summary, mapping)

        return {
            "round": round_num,
            "responses": responses,
            "errors": error_results,
            "aggregate": aggregate,
            "rapporteur": {"model": rapporteur_model, "summary": rapporteur_summary} if rapporteur_model else None,
        }

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    def _render_markdown_round(self, round_data: Dict[str, Any], method: str, prompt: str) -> str:
        """Render a single round's structured data as Markdown/plain text."""
        if "error" in round_data:
            return f"Error: {round_data['error']}"

        responses = round_data.get("responses", [])
        errors = round_data.get("errors", [])
        aggregate = round_data.get("aggregate") or {}
        rapporteur = round_data.get("rapporteur")
        text = ""

        if method in ("VoteYesNo", "VoteOptions"):
            votes = aggregate.get("votes", {})
            winner = aggregate.get("winner", "No Majority")
            threshold = aggregate.get("threshold", 0.5)
            result_summary = f"Votes: {votes}. Winner: {winner} (Threshold: >{threshold:.0%})"
            breakdown = "\n".join(
                f" - {r['model']}: {r['response']} (Confidence: {r['confidence_score']:.2f})"
                for r in responses
            )
            text = f"Result: {winner}\nDetails: {result_summary}\n\nVote Breakdown:\n{breakdown}"
            if rapporteur:
                text += f"\n\nRapporteur ({rapporteur['model']}) Summary:\n{rapporteur['summary']}"

        elif method == "Probability":
            if aggregate.get("abstained"):
                text = "Result: All LLMs abstained."
            elif "error" in aggregate:
                text = f"Error: {aggregate['error']}"
            else:
                avg = aggregate.get("average", 0)
                med = aggregate.get("median", 0)
                lo = aggregate.get("min", 0)
                hi = aggregate.get("max", 0)
                result_summary = f"Average: {avg:.2f} (Median: {med:.2f}, Range: {lo:.2f} - {hi:.2f})"
                text = f"Probability Analysis:\n{result_summary}"
                if rapporteur:
                    text += f"\n\nRapporteur ({rapporteur['model']}) Summary:\n{rapporteur['summary']}"

        elif method == "Compose":
            if "error" in aggregate:
                return f"Error: {aggregate['error']}"
            ranked = aggregate.get("ranked_candidates", [])
            lines = [f"Compose Results for: {prompt}", "=" * 40]
            for item in ranked:
                peer_lines = (
                    [
                        f"  * {pr['reviewer_model']}: {pr['score']} - {pr['justification']}"
                        for pr in item["peer_reviews"]
                    ]
                    if item["peer_reviews"]
                    else ["  (No ratings received)"]
                )
                lines += [
                    f"\nRank {item['rank']}: {item['model']} (as {item['pseudonym']})",
                    f"Average Score: {item['average_score']:.2f}",
                    "-" * 20,
                    f"Composition:\n{item['text']}",
                    "-" * 20,
                    "Peer Reviews:",
                    *peer_lines,
                    "=" * 40,
                ]
            unattr = aggregate.get("unattributed_reviewers") or []
            if unattr:
                lines.append(
                    f"\nNote: {len(unattr)} reviewer(s) returned ratings whose keys "
                    "could not be attributed to a candidate and were excluded: "
                    + ", ".join(u.get("reviewer_model", "?") for u in unattr)
                )
            text = "\n".join(lines)

        elif method == "Majority":
            text = f"Majority View Summary (by {rapporteur['model']}):\n{rapporteur['summary']}" if rapporteur else ""
        elif method == "Consensus":
            text = f"Consensus/Common Ground (by {rapporteur['model']}):\n{rapporteur['summary']}" if rapporteur else ""
        elif method == "Minority":
            text = f"Minority/Gap Analysis (by {rapporteur['model']}):\n{rapporteur['summary']}" if rapporteur else ""
        elif method == "Synthesis":
            text = f"Synthesis (by {rapporteur['model']}):\n{rapporteur['summary']}" if rapporteur else ""

        # Append model status (fallbacks used + failures)
        status_lines = []
        for r in responses:
            if r.get("fallback_for"):
                status_lines.append(
                    f" - {r['fallback_for']}: [fallback] primary unavailable, used {r['model']} instead"
                )
        for e in errors:
            category = e.get("error_category", "error")
            fallbacks = e.get("attempted_fallbacks")
            if fallbacks:
                chain = " → ".join(fallbacks)
                status_lines.append(f" - {chain}: [{category}] all options failed — {e['error']}")
            else:
                status_lines.append(f" - {e['model']}: [{category}] {e['error']}")
        if status_lines:
            text += "\n\n--- Model Status ---\n" + "\n".join(status_lines)

        return text

    def _render_markdown(self, result: Dict[str, Any]) -> str:
        """Convert a structured deliberation result dict to Markdown/plain text."""
        if "error" in result and "rounds" not in result:
            return f"Error: {result['error']}"

        method = result["method"]
        prompt = result["prompt"]
        rounds = result["rounds"]

        rendered = [self._render_markdown_round(r, method, prompt) for r in rounds]

        if len(rendered) == 1:
            return rendered[0]

        return (
            f"Pre-Deliberation Results:\n{rendered[0]}\n\n"
            f"{'=' * 40}\n\n"
            f"Post-Deliberation Results:\n{rendered[1]}"
        )

    # ------------------------------------------------------------------
    # Core deliberation engine
    # ------------------------------------------------------------------

    async def _deliberate(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        selected_llms: Optional[List[Slot]],
        method: str,
        method_options: Optional[Dict[str, Any]],
        deliberative: bool,
        api_keys: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        show_real_names_in_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Core deliberation engine. Returns a structured, JSON-serialisable result dict.
        On validation or quorum failures, returns a dict with a top-level 'error' key.
        """
        self._api_keys = api_keys
        if not user_prompt or not user_prompt.strip():
            return {"error": "user_prompt must not be empty."}

        if method_options is None:
            method_options = {}

        if not selected_llms:
            selected_llms = self.llms

        if not selected_llms:
            return {"error": "No LLMs configured. Add models to config or pass selected_llms."}

        for slot in selected_llms:
            for model in ([slot] if isinstance(slot, str) else slot):
                try:
                    self._validate_model(model)
                except ValueError as e:
                    return {"error": str(e)}

        min_models = self.config.get("defaults", {}).get("min_models", 1)
        sys_base = self.prompts["system_base"]
        sys_full = f"{system_prompt}\n\n{sys_base}" if system_prompt else sys_base
        lang_directive = self._language_directive(language)
        if lang_directive:
            sys_full = f"{lang_directive}\n\n{sys_full}"

        try:
            base_prompt = self._prepare_base_prompt(method, user_prompt, method_options)
        except ValueError as e:
            return {"error": str(e)}

        results1, errors1 = await self._gather_responses(selected_llms, sys_full, base_prompt)

        if len(results1) < min_models:
            return {
                "error": (
                    f"Quorum not met — {len(results1)} of {len(selected_llms)} model(s) responded "
                    f"(minimum required: {min_models})."
                ),
                "failed_models": [
                    {"model": e["model"], "error_category": e.get("error_category", "error"), "error": e["error"]}
                    for e in errors1
                ],
            }

        round1 = await self._build_round_data(
            1, method, user_prompt, method_options, results1, errors1,
            language=language, show_real_names_in_report=show_real_names_in_report,
        )

        result: Dict[str, Any] = {
            "schema_version": "1.0",
            "method": method,
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "deliberative": deliberative,
            "models": [_primary(slot) for slot in selected_llms],
            "rounds": [round1],
        }

        if not deliberative:
            return result

        # Round 2 — Deliberative: use round 1 pseudonyms for the peer-response context
        prompts_round2: Dict[str, str] = {}
        for slot in selected_llms:
            primary = _primary(slot)
            other_responses = [
                f"{r['pseudonym']}: {r['response']} (Reason: {r['reason']})"
                for r in round1["responses"]
                if r["model"] != primary
            ]
            if not other_responses:
                prompts_round2[primary] = base_prompt
            else:
                peer_text = "\n".join(other_responses)
                prompts_round2[primary] = self.prompts.get("deliberative_instruction", "").format(
                    peer_responses=peer_text,
                    original_prompt=base_prompt,
                )

        results2, errors2 = await self._gather_responses(selected_llms, sys_full, prompts_round2)

        if len(results2) < min_models:
            result["rounds"].append({
                "round": 2,
                "error": (
                    f"Quorum not met — {len(results2)} of {len(selected_llms)} model(s) responded "
                    f"(minimum required: {min_models})."
                ),
                "errors": errors2,
            })
            return result

        round2 = await self._build_round_data(
            2, method, user_prompt, method_options, results2, errors2,
            language=language, show_real_names_in_report=show_real_names_in_report,
        )

        # Reveal real names in round-2 free-text fields (response, reason) and
        # in the round-2 rapporteur summary using round-1 pseudonyms.
        # Round-2 LLMs saw peers under round-1 pseudonyms, so round-1 IDs leak
        # into their reason/response and — via the anonymised text built from
        # those fields — into the rapporteur's summary. This second pass is
        # output-only; round-2 prompts were sent with pseudonyms intact,
        # preserving blind deliberation.
        if show_real_names_in_report:
            round1_mapping = {
                r["model"]: r["pseudonym"]
                for r in round1.get("responses", [])
                if r.get("pseudonym")
            }
            for resp in round2.get("responses", []):
                resp["response"] = _reveal_names_in_text(
                    str(resp.get("response", "")), round1_mapping
                )
                resp["reason"] = _reveal_names_in_text(
                    str(resp.get("reason", "")), round1_mapping
                )
            rapporteur2 = round2.get("rapporteur") or {}
            if rapporteur2.get("summary"):
                rapporteur2["summary"] = _reveal_names_in_text(
                    rapporteur2["summary"], round1_mapping
                )

        result["rounds"].append(round2)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        selected_llms: Optional[List[Slot]] = None,
        method: str = 'VoteYesNo',
        method_options: Dict[str, Any] = None,
        deliberative: bool = False,
        api_keys: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        show_real_names_in_report: bool = True,
    ) -> str:
        """Run deliberation and return a Markdown-formatted report string."""
        result = await self._deliberate(
            user_prompt, system_prompt, selected_llms, method, method_options, deliberative,
            api_keys=api_keys, language=language,
            show_real_names_in_report=show_real_names_in_report,
        )
        return self._render_markdown(result)

    async def run_structured(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        selected_llms: Optional[List[Slot]] = None,
        method: str = 'VoteYesNo',
        method_options: Dict[str, Any] = None,
        deliberative: bool = False,
        api_keys: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        show_real_names_in_report: bool = True,
    ) -> Dict[str, Any]:
        """Run deliberation and return a structured, JSON-serialisable result dict."""
        return await self._deliberate(
            user_prompt, system_prompt, selected_llms, method, method_options, deliberative,
            api_keys=api_keys, language=language,
            show_real_names_in_report=show_real_names_in_report,
        )
