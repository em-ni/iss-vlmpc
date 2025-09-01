import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to save segmented image next to original image
def save_segmented_image(original_image, segmented_image_tensor, output_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    original_image.save(output_path.replace(".png", "_original.png"))
    
    # The mask is a boolean tensor, we convert it to uint8 and multiply by 255
    mask_image = (segmented_image_tensor.cpu().numpy()).astype('uint8') * 255
    
    # Debug: print shape
    print(f"Mask shape: {mask_image.shape}")
    
    # Ensure the mask is 2D by taking the first 2D slice if it's 3D
    while mask_image.ndim > 2:
        mask_image = mask_image[0] if mask_image.shape[0] == 1 else mask_image.squeeze()
        if mask_image.ndim > 2:
            mask_image = mask_image[0]
    
    print(f"Final mask shape: {mask_image.shape}")
    
    # Create a colored overlay like the SAM demo
    import numpy as np
    
    # Convert original image to numpy array
    original_np = np.array(original_image)
    
    # Create a colored mask (blue overlay)
    colored_mask = np.zeros_like(original_np)
    colored_mask[:, :, 2] = mask_image  # Blue channel
    
    # Create the overlay by blending original image with colored mask
    alpha = 0.5  # Transparency
    overlay = original_np.copy()
    mask_bool = mask_image > 0
    overlay[mask_bool] = (alpha * original_np[mask_bool] + (1 - alpha) * colored_mask[mask_bool]).astype(np.uint8)
    
    # Save both the binary mask and the colored overlay
    binary_mask = Image.fromarray(mask_image, mode='L')
    binary_mask.save(output_path.replace(".png", "_mask.png"))
    
    colored_overlay = Image.fromarray(overlay)
    colored_overlay.save(output_path.replace(".png", "_segmented.png"))

def save_all_masks(original_image, masks_list, scores, output_path_base):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path_base), exist_ok=True)
    original_image.save(output_path_base.replace(".png", "_original.png"))
    
    import numpy as np
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # RGB colors
    
    print(f"Found {len(masks_list)} masks")
    
    for i, mask_tensor in enumerate(masks_list):
        # Process mask
        mask_image = (mask_tensor.cpu().numpy()).astype('uint8') * 255
        
        # Ensure the mask is 2D
        while mask_image.ndim > 2:
            mask_image = mask_image[0] if mask_image.shape[0] == 1 else mask_image.squeeze()
            if mask_image.ndim > 2:
                mask_image = mask_image[0]
        
        # Convert original image to numpy array
        original_np = np.array(original_image)
        
        # Create a colored mask with different color for each mask
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(original_np)
        colored_mask[:, :, 0] = (mask_image * color[0] / 255).astype(np.uint8)
        colored_mask[:, :, 1] = (mask_image * color[1] / 255).astype(np.uint8) 
        colored_mask[:, :, 2] = (mask_image * color[2] / 255).astype(np.uint8)
        
        # Create the overlay
        alpha = 0.5
        overlay = original_np.copy()
        mask_bool = mask_image > 0
        overlay[mask_bool] = (alpha * original_np[mask_bool] + (1 - alpha) * colored_mask[mask_bool]).astype(np.uint8)
        
        # Get score for this mask
        if i < len(scores[0]):
            score = scores[0][i].item() if scores[0][i].numel() == 1 else scores[0][i].max().item()
        else:
            score = 0.0
        
        # Save individual mask
        binary_mask = Image.fromarray(mask_image, mode='L')
        binary_mask.save(output_path_base.replace(".png", f"_mask_{i}_score_{score:.3f}.png"))
        
        colored_overlay = Image.fromarray(overlay)
        colored_overlay.save(output_path_base.replace(".png", f"_segmented_{i}_score_{score:.3f}.png"))
        
        print(f"Saved mask {i} with score {score:.3f}")

# SAM
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Get raw masks before post-processing to see all proposals
raw_masks = outputs.pred_masks.cpu()
raw_scores = outputs.iou_scores.cpu()

print(f"Raw masks shape: {raw_masks.shape}")
print(f"Raw scores shape: {raw_scores.shape}")

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

# Save all masks instead of just the best one
# Use raw masks to get all proposals
raw_masks_resized = []
for i in range(raw_masks.shape[1]):
    mask_resized = processor.image_processor.post_process_masks(
        raw_masks[:, i:i+1], inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    raw_masks_resized.extend(mask_resized[0])

save_all_masks(raw_image, raw_masks_resized, raw_scores, "sam_data/car.png")

# SAM with segmentation map
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Get raw masks before post-processing to see all proposals
raw_masks = outputs.pred_masks.cpu()
raw_scores = outputs.iou_scores.cpu()

print(f"Raw masks shape: {raw_masks.shape}")
print(f"Raw scores shape: {raw_scores.shape}")

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

# Save all masks instead of just the best one
# Use raw masks to get all proposals
raw_masks_resized = []
for i in range(raw_masks.shape[1]):
    mask_resized = processor.image_processor.post_process_masks(
        raw_masks[:, i:i+1], inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    raw_masks_resized.extend(mask_resized[0])

save_all_masks(raw_image, raw_masks_resized, raw_scores, "sam_data/car2.png")

# SlimSAM
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77").to("cuda")
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]] # 2D localization of a window
inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to("cuda")
outputs = model(**inputs)

# Get raw masks before post-processing to see all proposals
raw_masks = outputs.pred_masks.cpu()
raw_scores = outputs.iou_scores.cpu()

print(f"Raw masks shape: {raw_masks.shape}")
print(f"Raw scores shape: {raw_scores.shape}")

masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

# Save all masks instead of just the best one
# Use raw masks to get all proposals
raw_masks_resized = []
for i in range(raw_masks.shape[1]):
    mask_resized = processor.image_processor.post_process_masks(
        raw_masks[:, i:i+1], inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    raw_masks_resized.extend(mask_resized[0])

save_all_masks(raw_image, raw_masks_resized, raw_scores, "sam_data/car3.png")
