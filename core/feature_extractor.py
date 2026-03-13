import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        """
        Initializes a modified ResNet-18 model for feature extraction.
        The final classification layer is removed to output the 512-D feature vector.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer to get 512-D features
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.model.eval()
        self.model.to(self.device)
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, crop):
        """
        Takes an image crop (numpy array from cv2) and returns a 512-D feature vector.
        """
        if crop is None or crop.size == 0:
            return None
            
        # Convert BGR (OpenCV) to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            
        # Flatten the (1, 512, 1, 1) tensor to (512,)
        feature_vector = features.squeeze().cpu().numpy()
        
        # L2 Normalize the vector for cosine similarity comparison later
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector
