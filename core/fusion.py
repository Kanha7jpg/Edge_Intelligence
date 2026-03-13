import numpy as np

class FeatureFusion:
    def __init__(self, alpha=0.3):
        """
        Initializes the fusion engine.
        :param alpha: The weight given to the *new* feature vector. 
                      Higher alpha adapts faster to new appearances but is more sensitive to noise.
        """
        self.alpha = alpha

    def fuse(self, existing_vector, new_vector):
        """
        Fuses an incoming feature vector with an existing one using a running average.
        Both vectors should be 1D numpy arrays of the same shape.
        """
        if existing_vector is None:
            return new_vector
        
        # Exponential moving average
        fused = (1.0 - self.alpha) * existing_vector + self.alpha * new_vector
        
        # Re-normalize to unit length after fusion
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
            
        return fused
