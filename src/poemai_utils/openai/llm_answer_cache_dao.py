import logging
from abc import ABC, abstractmethod

import sqlitedict
from poemai_utils.configuration_api import ConfigurationAPI

_logger = logging.getLogger(__name__)


class LLMAnswerCacheDao(ABC):
    @abstractmethod
    def store_llm_answer(self, prompt, answer, prompt_hash, answer_info_dict):
        raise NotImplementedError

    @abstractmethod
    def get_llm_answer(self, prompt_hash):
        raise NotImplementedError


class LLMAnswerCacheSqliteDict(LLMAnswerCacheDao):
    def __init__(self, config: ConfigurationAPI):
        self.cache = sqlitedict(
            config.LLM_ANSWER_CACHE_SQLITE_DICT_PATH, autocommit=True
        )

    def store_llm_answer(self, prompt_hash, answer_info_dict):
        self.cache[prompt_hash] = answer_info_dict

        return prompt_hash

    def get_llm_answer(self, prompt_hash):
        return self.cache[prompt_hash]


class LLMAnswerCacheDaoDict(LLMAnswerCacheDao):
    def __init__(self):
        self.cache = {}

    def store_llm_answer(self, prompt_hash, answer_info_dict):
        self.cache[prompt_hash] = answer_info_dict

        return prompt_hash

    def get_llm_answer(self, prompt_hash):
        return self.cache.get(prompt_hash, None)
