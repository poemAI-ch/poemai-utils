from collections import deque


class SimpleCache:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = {}
        self.queue = deque()

    def put(self, key, data):
        if key in self.cache:
            self.cache[key] = data
            self.queue.remove(key)
            self.queue.append(key)
            return
        if len(self.cache) >= self.max_size:
            old_key = self.queue.popleft()
            del self.cache[old_key]
        self.cache[key] = data
        self.queue.append(key)

    def get(self, key):
        return self.cache.get(key, None)
