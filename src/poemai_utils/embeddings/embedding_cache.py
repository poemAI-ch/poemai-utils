import logging
from hashlib import sha256

import numpy as np
from sqlitedict import SqliteDict

_logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self):
        self.cache = None

    def get(self, text, model_name):
        text_hash = self.text_hash(text)
        key = self.calc_key(model_name, text_hash)
        short_key = (model_name, text_hash[:6])
        if key in self.cache:
            _logger.debug(f"Cache hit for {short_key}")
            return np.array(self.cache[key])
        return None

    def put(self, text, embedding, model_name):
        text_hash = self.text_hash(text)
        key = self.calc_key(model_name, text_hash)
        self.cache[key] = embedding.tolist()

    def text_hash(self, text):
        # calculate sha256
        return sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def calc_key(model_name, text_hash):
        return model_name + "_" + text_hash


class EmbeddingCacheSqliteDict(EmbeddingCache):
    def __init__(self, cache_file):
        self.cache = SqliteDict(cache_file, autocommit=True)

    def get(self, text, model_name):
        text_hash = self.text_hash(text)
        key = self.calc_key(model_name, text_hash)
        short_key = (model_name, text_hash[:6])
        if key in self.cache:
            _logger.debug(f"Cache hit for {short_key}")
            return np.array(self.cache[key])
        return None


class EmbeddingCacheMemory(EmbeddingCache):
    def __init__(self):
        self.cache = {}
