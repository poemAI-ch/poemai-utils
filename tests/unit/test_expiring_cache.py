import time

from poemai_utils.expiring_cache import ExpiringCache


def test_expiring_cache_max_items():

    cache = ExpiringCache(max_size=2, expiry_time_seconds=600)
    cache.put("key1", "data1")
    cache.put("key2", "data2")
    cache.put("key3", "data3")

    assert cache.get("key1") is None
    assert cache.get("key2") == "data2"
    assert cache.get("key3") == "data3"


def test_expiring_cache_timeout():

    expiry_time_seconds = 0.1
    cache = ExpiringCache(max_size=2, expiry_time_seconds=expiry_time_seconds)
    cache.put("key1", "data1")
    time.sleep(0.5 * expiry_time_seconds)
    cache.put("key2", "data2")
    assert cache.get("key1") == "data1"
    assert cache.get("key2") == "data2"
    time.sleep(0.8 * expiry_time_seconds)
    assert cache.get("key1") is None
    assert cache.get("key2") == "data2"
    time.sleep(1.5 * expiry_time_seconds)
    assert cache.get("key1") is None
    assert cache.get("key2") is None
