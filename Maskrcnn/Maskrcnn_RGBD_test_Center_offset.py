import os
import json
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.cm as cm

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, annotation_json, rgb_dir, depth_dir):
        with open(annotation_json, 'r') as f:
            self.coco_data = json.load(f)
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir

    def __len__(self):
        return len(self.coco_data['images'])

    def __getitem__(self, idx):
        image_info = self.coco_data['images'][idx]
        image_file_name = image_info['file_name']

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, image_file_name)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_img).astype(np.float32)

        # Load corresponding depth image
        depth_path = os.path.join(self.depth_dir, image_file_name.replace('.jpg', '.png'))
        depth_img = Image.open(depth_path).convert("L")  # Convert to grayscale
        depth_np = np.array(depth_img).astype(np.float32)

        # Ensure depth has a channel dimension
        if len(depth_np.shape) == 2:
            depth_np = np.expand_dims(depth_np, axis=2)
        combined_img = np.concatenate([rgb_np, depth_np], axis=2)

        # Prepare the RGB-D image
        rgbd_img = torch.tensor(combined_img, dtype=torch.float32).permute(2, 0, 1)  # Convert to [C, H, W]

        # Get ground truth annotations
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_info['id']]
        masks = []
        for ann in annotations:
            segmentation = ann['segmentation']
            mask = self.coco_segmentation_to_mask(segmentation, image_info['width'], image_info['height'])
            masks.append(torch.tensor(mask, dtype=torch.uint8))

        # Stack ground truth masks to create a tensor
        gt_masks = torch.stack(masks).squeeze(1) if masks else None

        return rgbd_img, gt_masks  # Return RGBD image and ground truth masks

    @staticmethod
    def coco_segmentation_to_mask(segmentation, width, height):
        from pycocotools import mask as coco_mask
        mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(segmentation, list):
            for polygon in segmentation:
                if isinstance(polygon, list):
                    flattened_polygon = np.array(polygon).flatten().tolist()
                    rles = coco_mask.frPyObjects([flattened_polygon], height, width)
                    rle = coco_mask.merge(rles)
                    mask += coco_mask.decode(rle)
        elif isinstance(segmentation, dict):
            mask = coco_mask.decode(segmentation)
        else:
            raise TypeError(f"Invalid segmentation format: {type(segmentation)}")

        return mask

# Load Model Function
def load_model(weights_path):
    model = maskrcnn_resnet50_fpn(pretrained=False)  # Initialize model without pretrained weights
    model = modify_maskrcnn_for_rgbd(model)  # Modify for RGB-D input
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))  # Load model weights
    model.eval()
    return model

# Modify Mask R-CNN for RGB-D
def modify_maskrcnn_for_rgbd(model):
    in_channels = 4  # Change from 3 (RGB) to 4 (RGB + Depth)
    original_backbone = model.backbone.body
    original_backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.transform.image_mean = [0, 0, 0, 0]  # Mean for RGBD
    model.transform.image_std = [1, 1, 1, 1]    # Std for RGBD
    return model

# Function to Run Inference
def run_inference(model, rgbd_img, device, confidence_threshold=0.5):
    model.eval()
    rgbd_img = rgbd_img.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        predictions = model(rgbd_img)

    final_predictions = []
    for prediction in predictions:
        scores = prediction['scores'].cpu().numpy()
        high_conf_idx = np.where(scores >= confidence_threshold)[0]

        if len(high_conf_idx) > 0:
            final_prediction = {
                'boxes': prediction['boxes'][high_conf_idx].cpu().numpy(),
                'labels': prediction['labels'][high_conf_idx].cpu().numpy(),
                'masks': prediction['masks'][high_conf_idx].cpu().numpy(),
                'scores': prediction['scores'][high_conf_idx].cpu().numpy(),
            }
            final_predictions.append(final_prediction)

    return final_predictions



import cv2  # Ensure you have OpenCV installed

def calculate_center_opencv(mask):
    # Find the coordinates of non-zero mask pixels
    y_indices, x_indices = np.where(mask > 0)  # (row, column) = (y, x)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No pixels in the mask
    center_x = int(np.mean(x_indices))  # X coordinate
    center_y = int(np.mean(y_indices))  # Y coordinate
    return (center_x, center_y)  # Return as (x, y)

def visualize_results(rgbd_img, predictions, gt_masks):
    # Convert the RGB-D image to NumPy for visualization
    rgb_img_np = rgbd_img[:3].permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Get the RGB channels

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img_np)
    ax = plt.gca()

    blended_image_predictions = rgb_img_np.copy()  
    pred_centers = []  # Store predicted centers
    gt_centers = []    # Store ground truth centers

    # Overlay predicted masks on the image
    for pred in predictions:
        boxes = pred['boxes']
        masks = pred['masks']
        scores = pred['scores']

        for j in range(len(masks)):
            mask = masks[j, 0]  # Assuming mask shape [1, H, W]
            mask_np = mask > 0.5  # Create binary mask

            # Generate a color for the predicted mask
            color = cm.viridis(j / len(masks))  # Use colormap
            color_mask = np.zeros_like(rgb_img_np)
            color_mask[mask_np] = [color[0] * 255, color[1] * 255, color[2] * 255]  # Set the mask color

            # Blend the predicted mask with the blended image
            blended_image_predictions = np.where(color_mask > 0, color_mask, blended_image_predictions)

            # Draw bounding box
            box = boxes[j]
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f'{scores[j]:.2f}', color='white', backgroundcolor='black', fontsize=10)

            # Calculate and store the predicted center
            pred_center = calculate_center_opencv(mask_np)
            if pred_center:
                pred_centers.append(pred_center)
                ax.plot(pred_center[0], pred_center[1], 'o', color='red', markersize=10)  # Predicted center
                ax.text(pred_center[0], pred_center[1], f'({pred_center[0]}, {pred_center[1]})', color='red', fontsize=8)

    # Overlay ground truth centers
    if gt_masks is not None:
        for i in range(gt_masks.shape[0]):
            gt_mask_np = gt_masks[i].cpu().numpy()
            gt_mask_binary = gt_mask_np > 0  # Create binary mask for ground truth

            # Calculate and store the ground truth center
            gt_center = calculate_center_opencv(gt_mask_np)
            if gt_center:
                gt_centers.append(gt_center)
                ax.plot(gt_center[0], gt_center[1], 'o', color='blue', markersize=10)  # Ground truth center
                ax.text(gt_center[0], gt_center[1], f'({gt_center[0]}, {gt_center[1]})', color='blue', fontsize=8)

    # Display blended predicted masks
    plt.imshow(blended_image_predictions)  
    plt.title("Predicted Masks with Bounding Boxes and Centers")
    plt.axis('off')
    
    blended_image_gt = rgb_img_np.copy()  # Create a copy of the original RGB image for GT overlay
    print("Shape of gt_masks:", gt_masks.shape[0])
    if gt_masks is not None:  # Check if gt_masks contains 'masks'
        # Assuming gt_masks is a tensor of shape [N, H, W]

        for i in range(gt_masks.shape[0]):  # Access directly if gt_masks is a tensor
            gt_mask_np = gt_masks[i].cpu().numpy()  # This should work if gt_masks is a tensor
            gt_mask_binary = gt_mask_np > 0  # Create binary mask for ground truth

            # Assign a color for ground truth (e.g., blue)
            color_mask = np.zeros_like(rgb_img_np)  # Create an empty mask
            color_mask[gt_mask_binary] = [0, 0, 255]  # Blue color for ground truth

            # Blend the ground truth mask with the blended image
            blended_image_gt = np.where(color_mask > 0, color_mask, blended_image_gt)



    # Show the blended image for ground truth masks
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image_gt)  
    plt.title("Ground Truth Masks")
    plt.axis('off')

    plt.show()
    
    
    

    # Calculate distances from GT centers to predicted centers
    distances = []  # List to store distances
    for gt_center in gt_centers:
        # Calculate distances to all predicted centers
        dist_to_pred_centers = [np.sqrt((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2) for pred_center in pred_centers]
        
        # Find the minimum distance and corresponding predicted center
        if dist_to_pred_centers:
            min_distance = min(dist_to_pred_centers)
            min_distance_index = dist_to_pred_centers.index(min_distance)
            closest_pred_center = pred_centers[min_distance_index]

            # Draw a line between the GT center and the closest predicted center
            plt.plot([gt_center[0], closest_pred_center[0]], [gt_center[1], closest_pred_center[1]], 'k--')  # Line between centers
            mid_point = ((gt_center[0] + closest_pred_center[0]) / 2, (gt_center[1] + closest_pred_center[1]) / 2)
            plt.text(mid_point[0], mid_point[1], f'{min_distance:.2f}', fontsize=8, ha='center')

            # Store the distance in the list
            distances.append((gt_center, closest_pred_center, min_distance))

    plt.show()
    if distances:
        mean_distance = np.mean([dist for _, _, dist in distances])  # Extract distances
        print(f"Mean Distance: {mean_distance:.2f} pixels")
    else:
        print("No distances to calculate mean.")
    
    # Print all calculated distances
    print("Calculated Distances from GT Centers to Closest Predicted Centers:")
    for gt_center, pred_center, dist in distances:
        print(f"GT Center: {gt_center}, Closest Pred Center: {pred_center}, Distance: {dist:.2f} pixels")
    

    return distances  
    






# Main Function remains unchanged, just replace the visualize_results call


# Main Function
def main():
    # Define your paths
    annotation_json = '/home/pouyas/test/json/test.json'  # Path to your validation JSON
    rgb_dir = '/home/pouyas/test/rgb'  # Path to RGB images
    depth_dir = '/home/pouyas/test/depth'  # Path to depth images
    weights_path = '/home/pouyas/best_RGBD_model_with_RGB_weights_fold_4_epoch_61_iou_0.3831.pth'  # Path to model weights

    # Load dataset and model
    dataset = CustomDataset(annotation_json, rgb_dir, depth_dir)
    model = load_model(weights_path)  # Load and modify the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mean_distances = [] 

    # Test on random images from the dataset
    for I in range(14):  # Test on 5 random images
        image_idx = random.randint(0, len(dataset) - 1)
        rgbd_img, gt_masks = dataset[I]  # Get RGB-D image and ground truth masks

        predictions = run_inference(model, rgbd_img, device, confidence_threshold=0.9)
        distances = visualize_results(rgbd_img, predictions, gt_masks)
        if distances:
           
            mean_distance = np.mean([dist for _, _, dist in distances])  # Extract distances
            mean_distances.append(mean_distance)  # Add to the list
        #visualize_results(rgbd_img, predictions, gt_masks)
        print("Mean Distances for All Images:")
        for idx, mean in enumerate(mean_distances):
            print(f"Image {idx + 1}: Mean Distance: {mean:.2f} pixels")


# Run the main function
if __name__ == "__main__":
    main()
