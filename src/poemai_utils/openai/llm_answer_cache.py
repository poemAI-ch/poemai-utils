import hashlib
import json
from decimal import Decimal

from poemai_utils.openai.llm_answer_cache_dao import LLMAnswerCacheDao


class LLMAnswerCache:
    def __init__(self, llm_answer_cache_dao: LLMAnswerCacheDao):
        self.llm_answer_cache_dao = llm_answer_cache_dao

    def fetch_from_cache(
        self,
        model,
        prompt,
        temperature,
        max_tokens,
        stop,
        suffix,
        system_prompt,
        messages,
    ):
        """Fetch an answer from the cache."""
        cache_key = self.calc_cache_key(
            model,
            prompt,
            temperature,
            max_tokens,
            stop,
            suffix,
            system_prompt,
            messages,
        )
        answer = self.llm_answer_cache_dao.get_llm_answer(cache_key)
        if answer is not None:
            return answer["answer"], cache_key
        return answer, cache_key

    def store_in_cache(
        self,
        model,
        prompt,
        temperature,
        max_tokens,
        stop,
        suffix,
        system_prompt,
        messages,
        answer,
    ):
        """Store an answer in the cache."""
        cache_key = self.calc_cache_key(
            model,
            prompt,
            temperature,
            max_tokens,
            stop,
            suffix,
            system_prompt,
            messages,
        )
        answer_info_dict = {
            "model": model,
            "prompt": prompt,
            "temperature": Decimal(temperature),
            "max_tokens": max_tokens,
            "stop": stop,
            "suffix": suffix,
            "system_prompt": system_prompt,
            "messages": messages,
            "answer": answer,
        }

        self.llm_answer_cache_dao.store_llm_answer(cache_key, answer_info_dict)

    @staticmethod
    def calc_cache_key(
        model,
        prompt,
        temperature,
        max_tokens,
        stop,
        suffix,
        system_prompt,
        messages,
    ):
        # calculate sha256 hash of prompt
        sha256 = hashlib.sha256()
        if prompt is not None:
            sha256.update(prompt.encode("utf-8"))
        sha256.update(str(temperature).encode("utf-8"))
        sha256.update(str(max_tokens).encode("utf-8"))
        if stop is not None:
            sha256.update(str(stop).encode("utf-8"))
        if suffix is not None:
            sha256.update(str(suffix).encode("utf-8"))
        if system_prompt is not None:
            sha256.update(str(system_prompt).encode("utf-8"))
        if messages is not None:
            sha256.update(json.dumps(messages).encode("utf-8"))
        prompt_hash = sha256.hexdigest()
        return f"{model}_{prompt_hash}"
