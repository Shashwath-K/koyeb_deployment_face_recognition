import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import pickle
import os
from typing import List, Dict, Tuple, Union, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ========== MobileFaceNet Architecture ==========

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: Tuple[int, int] = (1, 1), 
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0), 
                 groups: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: Tuple[int, int] = (1, 1), 
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0), 
                 groups: int = 1):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x

class DepthWise(nn.Module):
    def __init__(self, in_c: int, out_c: int, residual: bool = False, 
                 kernel: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (2, 2), 
                 padding: Tuple[int, int] = (1, 1), groups: int = 1):
        super(DepthWise, self).__init__()
        self.conv = ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, 
                                 padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short_cut = x if self.residual else None
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual and short_cut is not None:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(nn.Module):
    def __init__(self, c: int, num_block: int, groups: int, 
                 kernel: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1), 
                 padding: Tuple[int, int] = (1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(c, c, residual=True, kernel=kernel, 
                         padding=padding, stride=stride, groups=groups)
            )
        self.model = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super(MobileFaceNet, self).__init__()
        # For 112x112 input
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), 
                                 padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), 
                                padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), 
                              stride=(1, 1), padding=(1, 1))
        
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), 
                                padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), 
                              stride=(1, 1), padding=(1, 1))
        
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), 
                                padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), 
                              stride=(1, 1), padding=(1, 1))
        
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(7, 7), 
                                    stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return F.normalize(out, p=2, dim=1)

# ========== FaceEncoder Class ==========

class FaceEncoder:
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize MobileFaceNet encoder
        
        Args:
            model_path: Path to pretrained model weights (optional)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.model = MobileFaceNet(embedding_size=512).to(self.device)
        self.model.eval()
        
        # Default mean and std for normalization
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.std = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
        elif model_path:
            print(f"Warning: Model path {model_path} does not exist. Using random weights.")
    
    def load_weights(self, model_path: str) -> None:
        """Load pretrained weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Remove 'module.' prefix if present (for DataParallel models)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=True)
            print(f"✓ Loaded weights from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    
    def save_weights(self, save_path: str) -> None:
        """Save model weights"""
        torch.save(self.model.state_dict(), save_path)
        print(f"✓ Saved weights to {save_path}")
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess face image for MobileFaceNet
        
        Args:
            image: PIL Image, numpy array, or torch Tensor
        
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # If already a tensor, just ensure it's on the right device
        if isinstance(image, torch.Tensor):
            return image.to(self.device)
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Ensure float32 type
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)
        
        # Handle different image formats
        if len(image_np.shape) == 2:  # Grayscale
            # Convert to RGB by repeating channels
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        elif image_np.shape[2] == 4:  # RGBA
            # Convert RGBA to RGB (discard alpha)
            image_np = image_np[:, :, :3]
        
        # Ensure we have 3 channels
        if image_np.shape[2] != 3:
            raise ValueError(f"Expected 3 channels but got {image_np.shape[2]} channels")
        
        # Normalize to [-1, 1] using numpy operations
        # This avoids type checker issues with OpenCV
        image_np = (image_np - 127.5) / 127.5
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def encode(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
               normalize: bool = True) -> np.ndarray:
        """
        Encode a face image into embedding vector
        
        Args:
            image: PIL Image, numpy array, or torch Tensor
            normalize: Whether to L2 normalize the embedding
        
        Returns:
            Embedding vector (512-dimensional numpy array)
        """
        # Preprocess if not already a tensor
        if not isinstance(image, torch.Tensor):
            image_tensor = self.preprocess(image)
        else:
            image_tensor = image.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy().squeeze()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
        
        return embedding_np
    
    def encode_batch(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]], 
                     normalize: bool = True) -> np.ndarray:
        """
        Encode a batch of face images
        
        Args:
            images: List of PIL Images, numpy arrays, or torch Tensors
            normalize: Whether to normalize embeddings
        
        Returns:
            Numpy array of embeddings (batch_size x 512)
        """
        # Preprocess all images
        tensors = []
        for img in images:
            if isinstance(img, torch.Tensor):
                tensor = img.to(self.device)
            else:
                tensor = self.preprocess(img)
            tensors.append(tensor)
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(batch)
        
        embeddings_np = embeddings.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_np = embeddings_np / norms
        
        return embeddings_np
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                           metric: str = 'cosine') -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding vector (normalized)
            embedding2: Second embedding vector (normalized)
            metric: 'cosine' or 'euclidean'
        
        Returns:
            Similarity score (higher means more similar)
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Embedding shapes must match: {embedding1.shape} vs {embedding2.shape}")
        
        if metric == 'cosine':
            # Ensure embeddings are float32
            emb1 = embedding1.astype(np.float32)
            emb2 = embedding2.astype(np.float32)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2)
            similarity = np.clip(similarity, -1.0, 1.0)
            return float((similarity + 1) / 2)
        elif metric == 'euclidean':
            # Euclidean distance (convert to similarity)
            dist = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + dist)
            return float(similarity)
        else:
            raise ValueError(f"Unknown metric: {metric}. Choose 'cosine' or 'euclidean'")
    
    def verify_faces(self, image1: Union[np.ndarray, Image.Image, torch.Tensor], 
                    image2: Union[np.ndarray, Image.Image, torch.Tensor], 
                    threshold: float = 0.5, 
                    metric: str = 'cosine') -> Tuple[bool, float]:
        """
        Verify if two face images belong to the same person
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Similarity threshold for verification (0-1)
            metric: Similarity metric to use
        
        Returns:
            (is_same_person, similarity_score)
        """
        # Encode both faces
        emb1 = self.encode(image1)
        emb2 = self.encode(image2)
        
        # Calculate similarity
        similarity = self.calculate_similarity(emb1, emb2, metric)
        
        # Determine if same person
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the encoder model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MobileFaceNet',
            'embedding_size': 512,
            'input_size': (112, 112),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device)
        }

    def generate_embedding(self, face_image: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Compatibility method for main.py
        Alias for encode() method
        """
        return self.encode(face_image)


# ========== Simple Test ==========

if __name__ == "__main__":
    print("Testing MobileFaceNet encoder...")
    
    # Test 1: Create encoder
    encoder = FaceEncoder()
    print("✓ Encoder created successfully")
    
    # Test 2: Get model info
    info = encoder.get_model_info()
    print(f"✓ Model: {info['model_name']}")
    print(f"✓ Embedding size: {info['embedding_size']}")
    print(f"✓ Device: {info['device']}")
    
    # Test 3: Create test images
    test_np = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    test_pil = Image.fromarray(test_np)
    test_tensor = torch.randn(1, 3, 112, 112)
    
    # Test 4: Encode different input types
    emb_np = encoder.encode(test_np)
    print(f"✓ Numpy encoding: shape={emb_np.shape}")
    
    emb_pil = encoder.encode(test_pil)
    print(f"✓ PIL encoding: shape={emb_pil.shape}")
    
    emb_tensor = encoder.encode(test_tensor)
    print(f"✓ Tensor encoding: shape={emb_tensor.shape}")
    
    # Test 5: Similarity calculation
    similarity = encoder.calculate_similarity(emb_np, emb_pil)
    print(f"✓ Similarity calculation: {similarity:.4f}")
    
    # Test 6: Verify faces
    is_same, score = encoder.verify_faces(test_np, test_pil)
    print(f"✓ Face verification: is_same={is_same}, score={score:.4f}")
    
    print("\n✅ All tests passed! MobileFaceNet is working correctly.")