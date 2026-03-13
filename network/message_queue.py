import queue
import threading

class ThreadSafeQueue:
    def __init__(self):
        """
        Initializes a thread-safe Queue for Store-and-Forward routing.
        Allows adding messages from main thread and popping from networking threads.
        """
        self.queue = queue.Queue()
        self.lock = threading.Lock()

    def enqueue(self, item):
        with self.lock:
            self.queue.put(item)

    def dequeue_all(self):
        """
        Atomically removes and returns all items currently in the queue.
        Useful for flushing the queue when a peer reconnects.
        """
        items = []
        with self.lock:
            while not self.queue.empty():
                items.append(self.queue.get())
        return items

    def is_empty(self):
        return self.queue.empty()
    
    def qsize(self):
        return self.queue.qsize()
