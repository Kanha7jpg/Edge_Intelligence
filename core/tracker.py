import numpy as np

class ReIDTracker:
    def __init__(self, similarity_threshold=0.65):
        """
        Initializes the Cosine Similarity Tracker.
        :param similarity_threshold: Minimum cosine similarity required to consider 
                                     two vectors as the same identity.
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.
        Assumes vectors are already L2-normalized.
        """
        return np.dot(vec1, vec2)

    def match(self, incoming_vector, memory_cache):
        """
        Compares an incoming vector against all identities in the LRU cache.
        Returns the best matched global_id and its similarity score.
        If no match exceeds the threshold, returns None.
        """
        best_match_id = None
        highest_sim = -1.0
        
        identities = memory_cache.get_all_identities()
        
        for global_id, data in identities.items():
            stored_vector = data['features']
            sim = self.compute_similarity(incoming_vector, stored_vector)
            
            if sim > highest_sim:
                highest_sim = sim
                best_match_id = global_id
                
        if highest_sim >= self.similarity_threshold:
            return best_match_id, highest_sim
        else:
            return None, highest_sim
