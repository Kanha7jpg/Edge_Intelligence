import hashlib
import json

class VectorCommitment:
    @staticmethod
    def generate_commitment(feature_vector, global_id, timestamp):
        """
        Generates a SHA-256 hash for the given feature vector and metadata.
        This proves the integrity of the data being shared across the mesh.
        """
        data = {
            "global_id": global_id,
            "timestamp": timestamp,
            "vector": feature_vector.tolist()
        }
        json_data = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(json_data).hexdigest()

    @staticmethod
    def verify_commitment(feature_vector, global_id, timestamp, provided_hash):
        """
        Verifies that the generated hash matches the provided hash.
        """
        calculated_hash = VectorCommitment.generate_commitment(feature_vector, global_id, timestamp)
        return calculated_hash == provided_hash
