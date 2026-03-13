from collections import OrderedDict
import threading

class LRUMemory:
    def __init__(self, max_size=100):
        """
        Initializes the Least Recently Used (LRU) Cache.
        :param max_size: Maximum number of active tracking identities to hold.
        """
        self.max_size = max_size
        # Maps global_id -> {'features': np.array, 'last_seen': timestamp}
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, global_id):
        """
        Retrieves the identity data if it exists, marking it as recently used.
        """
        with self.lock:
            if global_id not in self.cache:
                return None
                
            # Move to end (most recently used)
            self.cache.move_to_end(global_id)
            return self.cache[global_id]

    def put(self, global_id, feature_vector, timestamp):
        """
        Adds or updates an identity in the cache. Evicts the oldest if full.
        """
        with self.lock:
            if global_id in self.cache:
                self.cache.move_to_end(global_id)
                
            self.cache[global_id] = {
                'features': feature_vector,
                'last_seen': timestamp
            }
            
            if len(self.cache) > self.max_size:
                # Pop the first item (least recently used)
                self.cache.popitem(last=False)

    def get_all_identities(self):
        """
        Returns a copy of all stored identities to prevent mutation errors during iteration.
        """
        with self.lock:
            return self.cache.copy()
