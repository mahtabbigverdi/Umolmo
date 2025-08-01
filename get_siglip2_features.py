import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModel
import numpy as np

# Load SigLIP-2 processor and model
model_id = "google/siglip2-large-patch16-256"  # Change if using another variant
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).cuda()

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))  # Resize to match model input size
    return image

def extract_patch_features(image):
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
    
    # Assuming 'patch embeddings' are in `last_hidden_state`, including [CLS]
    patch_embeddings = outputs.last_hidden_state  # [B, N, D]
    return patch_embeddings.squeeze(0)  # [N, D]

def interpolate_features(features, target_len):
    """
    Interpolate sequence of patch features to a given length.
    Args:
        features: torch.Tensor of shape [N, D]
        target_len: int, desired number of patches
    Returns:
        torch.Tensor of shape [target_len, D]
    """
    # N, D = features.shape
    # features = features.unsqueeze(0).permute(0, 2, 1)  # [1, D, N]
    # interpolated = F.interpolate(features, size=target_len, mode='linear', align_corners=False)
    # return interpolated.squeeze(0).permute(1, 0)  # [target_len, D]

    N, D = features.shape
    target_h = target_w = int(np.sqrt(target_len))              
    h = w = int(np.sqrt(N))
    image_features = features.view(h, w, D)  # [H, W, D]
    image_features = image_features.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    image_features = F.interpolate(image_features.to(torch.float32), size=(target_h, target_w), mode='bilinear', align_corners=False).to(image_features.dtype)
    image_features = image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
    return image_features.squeeze(0)  # [target_len, D]

def get_interpolated_patch_features(image_path, target_num_patches, save_path=None):
    image = load_image(image_path)

    patch_features = extract_patch_features(image)
    interpolated_features = interpolate_features(patch_features, target_num_patches)
    if save_path:
        np.save(save_path, interpolated_features.cpu().numpy())
    return interpolated_features  # shape: [target_num_patches, D]


