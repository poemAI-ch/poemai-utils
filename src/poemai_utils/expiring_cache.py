import time
from collections import deque


class ExpiringCache:
    def __init__(self, max_size=100, expiry_time_seconds=600):
        self.max_size = max_size
        self.cache = {}
        self.queue = deque()
        self.expiry_time_seconds = expiry_time_seconds

    def put(self, key, data):
        if key in self.cache:
            self.cache[key] = (data, time.time())
            self.queue.remove(key)
            self.queue.append(key)
            return
        if len(self.cache) >= self.max_size:
            old_key = self.queue.popleft()
            del self.cache[old_key]
        self.cache[key] = (data, time.time())
        self.queue.append(key)

    def get(self, key):
        cache_entry = self.cache.get(key, None)
        if cache_entry is None:
            return None

        data, timestamp = cache_entry
        if time.time() - timestamp > self.expiry_time_seconds:
            del self.cache[key]
            self.queue.remove(key)
            return None
        return data
