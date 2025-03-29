import torch
import torchvision

from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import os
import json
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision  # Add this line
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.ops import box_iou  # This is the function we will use to compute IoU


train_losses_per_epoch=[]
val_losses_per_epoch=[]
best_val_loss = float('inf')
import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, annotation_json, rgb_dir, depth_dir, transforms=None):
        with open(annotation_json, 'r') as f:
            self.coco_data = json.load(f)

        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.coco_data['images']) * 5  # Original + Augmented

    def __getitem__(self, idx):
        original_idx = idx // 5
        augmentation_idx = idx % 5

        image_info = self.coco_data['images'][original_idx]
        image_file_name = image_info['file_name']

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, image_file_name)
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_np = np.array(rgb_img).astype(np.float32)

        # Load corresponding depth image (assuming same name as RGB but in depth folder)
        depth_path = os.path.join(self.depth_dir, image_file_name.replace('.jpg', '.png'))
        depth_img = Image.open(depth_path).convert("L")  # Convert to grayscale (1 channel)
        depth_np = np.array(depth_img).astype(np.float32)
        if len(depth_np.shape) == 2:
            depth_np = np.expand_dims(depth_np, axis=2)

        # Concatenate RGB and depth to form RGB-D
        combined_img = np.concatenate([rgb_np, depth_np], axis=2)

        # Get target (annotations) for the image
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_info['id']]
        bboxes = []
        masks = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
            segmentation = ann['segmentation']
            mask = self.coco_segmentation_to_mask(segmentation, image_info['width'], image_info['height'])

            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            masks.append(torch.tensor(mask, dtype=torch.uint8))

        # Apply augmentations
        if augmentation_idx > 0 and self.transforms:
            if random.random() < 0.5:
                rgb_img = F.hflip(rgb_img)
                depth_img = F.hflip(depth_img)
                masks = [F.hflip(mask) for mask in masks]
                bboxes = self.flip_bboxes_horizontal(bboxes, rgb_img.size[0])

            if random.random() < 0.5:
                rgb_img = F.vflip(rgb_img)
                depth_img = F.vflip(depth_img)
                masks = [F.vflip(mask) for mask in masks]
                bboxes = self.flip_bboxes_vertical(bboxes, rgb_img.size[1])

        # Convert images to tensors
        rgb_np = np.array(rgb_img).astype(np.float32)  # Convert PIL Image to NumPy array
        rgb_img = torch.tensor(rgb_np, dtype=torch.float32).permute(2, 0, 1)  # Convert to Tensor
        depth_np = np.array(depth_img).astype(np.float32)  # Convert PIL Image to NumPy array
        depth_img = torch.tensor(depth_np, dtype=torch.float32).unsqueeze(0)  # Convert to Tensor and add a channel dimension


       



        # Prepare target
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks).squeeze(1)
        }

        # Concatenate depth channel to RGB image
        rgbd_img = torch.cat([rgb_img, depth_img], dim=0)

        # Visualize RGB, Depth, Masks, and Bboxes
       # self.visualize(rgb_img, depth_img, masks, bboxes, augmentation_idx)

        return rgbd_img, target

    @staticmethod
    def flip_bboxes_horizontal(bboxes, img_width):
        return [[img_width - x_max, y_min, img_width - x_min, y_max] for x_min, y_min, x_max, y_max in bboxes]

    @staticmethod
    def flip_bboxes_vertical(bboxes, img_height):
        return [[x_min, img_height - y_max, x_max, img_height - y_min] for x_min, y_min, x_max, y_max in bboxes]

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

    @staticmethod
    def visualize(rgb_img, depth_img, masks, bboxes, augmentation_idx):
        # Convert images to numpy arrays for visualization
        rgb_img_np = rgb_img.permute(1, 2, 0).numpy().astype(np.uint8)
        depth_img_np = depth_img.squeeze().numpy().astype(np.uint8)
        mask_combined = torch.stack(masks).sum(dim=0).squeeze().numpy()

        # Set up the plot
        plt.figure(figsize=(20, 10))

        # Plot RGB Image
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_img_np)
        plt.title(f"RGB Image (Aug {augmentation_idx})")
        plt.axis("off")

        # Plot Depth Image
        plt.subplot(1, 4, 2)
        plt.imshow(depth_img_np, cmap='gray')
        plt.title(f"Depth Image (Aug {augmentation_idx})")
        plt.axis("off")

        # Plot Masks
        plt.subplot(1, 4, 3)
        plt.imshow(mask_combined, cmap='gray')
        plt.title(f"Mask (Aug {augmentation_idx})")
        plt.axis("off")

        # Plot Bounding Boxes on RGB Image
        plt.subplot(1, 4, 4)
        plt.imshow(rgb_img_np)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='r', facecolor='none'))
        plt.title(f"RGB with Bboxes (Aug {augmentation_idx})")
        plt.axis("off")

        plt.show()

# Use this class and pass it into your DataLoader as before.
import torch
import torchvision

# Modify Mask R-CNN to accept RGB-D input


def load_rgb_weights_to_rgbd(rgbd_model, rgb_weights_path):
    # Load the RGB model weights
    rgb_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
    rgb_model.load_state_dict(torch.load(rgb_weights_path))

    # Get the state_dict from the RGB model
    rgb_state_dict = rgb_model.state_dict()

    # Get the state_dict from the RGB-D model
    rgbd_state_dict = rgbd_model.state_dict()

    # Filter out the first convolution layer's weights from the RGB model
    rgb_state_dict.pop('backbone.body.conv1.weight')

    # Update RGB-D model with RGB model weights
    rgbd_state_dict.update(rgb_state_dict)

    # Load the updated state_dict into the RGB-D model
    rgbd_model.load_state_dict(rgbd_state_dict)

    return rgbd_model

# Training Loop and other functions remain unchanged











# Use the same code for DataLoader, model setup, training loop, etc.
# Here is an example of DataLoader setup:

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))
# Modify Mask R-CNN to accept RGB-D input
def modify_maskrcnn_for_rgbd(model):
    # Modify the first convolution layer to accept 4 channels (RGBD)
    in_channels = 4  # Change from 3 (RGB) to 4 (RGB + Depth)
    original_backbone = model.backbone.body
    original_backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the normalization mean and std to handle 4 channels (adding for depth)
   # model.transform.image_mean = [0.485, 0.456, 0.406, 6.9]  # Add mean for the depth channel
   # model.transform.image_std = [0.229, 0.224, 0.225, 17.98]   # Add std for the depth channel
    model.transform.image_mean = [0.485, 0.456, 0.406,6.9]

    model.transform.image_std = [0.229, 0.224, 0.225,17.98]   # Add std for the depth channel

    return model

# Load Mask R-CNN model and modify it for RGBD input



# K-Fold Cross Validation training loop
def train_model_with_kfold(dataset, num_epochs=50, batch_size=2, k=5, device=torch.device('cuda')):
    """
    Trains a Mask R-CNN model using K-Fold Cross Validation (RGB-only) and saves the best model based on Mean IoU.
    
    Args:
        dataset: CustomDataset object containing the data.
        num_epochs: Number of epochs to train for each fold.
        batch_size: Batch size for training and validation.
        k: Number of folds for K-Fold cross validation.
        device: The device to train the model on (CPU or GPU).
    """
    # Initialize K-Fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Tracking Mean IoU for each epoch across all folds
    all_mean_ious = []  # For plotting IoU across all epochs
    best_iou = 0  # To track the best IoU overall

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold + 1}/{k}")

        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # DataLoaders for training and validation
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

        # Initialize a fresh Mask R-CNN model for each fold
        #model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
       # model.transform.image_mean = [0, 0, 0,0]  # Add mean for the depth channel
       # model.transform.image_std = [1,1,1,1]   # Add std for the depth channel

        #model.to(device)

        # Define optimizer
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0001, momentum=0.9, weight_decay=0.0005)

        # List to store mean IoU for each epoch in the current fold
        fold_mean_ious = []

        # Training loop for each fold
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

            # Evaluate the model on the validation fold after each epoch
            mean_iou = compute_mean_iou(model, val_loader, device)
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Mean IoU: {mean_iou}")
            fold_mean_ious.append(mean_iou)  # Track the Mean IoU for this epoch
            
            # Save the best model based on Mean IoU
            if mean_iou > best_iou:
                best_iou = mean_iou
                model_save_path = f'best_RGBD_model_with_RGB_weights_imagenet_normalization_fold_{fold + 1}_epoch_{epoch + 1}_iou_{best_iou:.4f}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved with IoU {best_iou:.4f} at {model_save_path}")

        # Add the Mean IoU for the fold to the overall tracking list
        all_mean_ious.extend(fold_mean_ious)  # This extends over the folds

    # After training all folds, plot the IoU vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(all_mean_ious) + 1), all_mean_ious, marker='o', label='Mean IoU')
    plt.title('Mean IoU vs Epoch')
    plt.xlabel('Epoch (Overall across 5 folds)')
    plt.ylabel('Mean IoU')
    plt.grid(True)
    plt.legend()
    plt.show()

    return all_mean_ious

# Mean IoU evaluation function
def compute_mean_iou(model, data_loader, device):
    model.eval()
    total_iou = 0
    total_boxes = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                true_boxes = targets[i]['boxes'].cpu()

                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    # Compute IoU for the predicted and ground truth boxes
                    ious = torchvision.ops.box_iou(pred_boxes, true_boxes)
                    total_iou += ious.mean().item()  # Average IoU for the current image
                    total_boxes += 1

    # Return the mean IoU across all images
    mean_iou = total_iou / total_boxes if total_boxes > 0 else 0
    return mean_iou

# Training Loop
def train_model(model, train_loader, val_loader, device, num_epochs=100):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0001, momentum=0.9, weight_decay=0.0005)
    best_iou = 0  # Initialize best IoU
    ious_per_epoch = []  # To store IoU per epoch

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

        # Compute mean IoU on validation set
        current_iou = compute_mean_iou(model, val_loader, device)
        ious_per_epoch.append(current_iou)  # Store IoU
        print(f"Epoch {epoch + 1}, Mean IoU: {current_iou}")

        # Save the model if it achieves the best IoU
        if current_iou > best_iou:
            best_iou = current_iou
            save_file = f'best_model_RGBD_COCO_epoch_{epoch + 1}_iou_{best_iou:.4f}.pth'
            torch.save(model.state_dict(), save_file)
            print(f"New best model saved: {save_file}")

    # Plot IoU per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ious_per_epoch) + 1), ious_per_epoch, marker='o', label='Mean IoU')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, marker='o', label='Training Loss')
    plt.title('Training Loss per Epoch  after Augmentation RGBD 750 data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)  # Initialize model without pretrained weights
    model = modify_maskrcnn_for_rgbd(model)  # Modify model for RGBD
   # weights_path='/home/pouyas/best_model_fold_4_epoch_89_iou_0.3713.pth'
    weights_path='/home/pouyas/best_model_RGB_Imagenet_normalization_fold_4_epoch_87_iou_0.3727.pth'
    
    # Load weights from RGB model into RGBD model
    model = load_rgb_weights_to_rgbd(model, weights_path)

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_dataset = CustomDataset(
        annotation_json='/home/pouyas/train.json',
        rgb_dir='/home/pouyas/rgb_image',
        depth_dir='/home/pouyas/depth_final',
        transforms=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_dataset = CustomDataset(
        annotation_json='/home/pouyas/val.json',
        rgb_dir='/home/pouyas/rgb_image',
        depth_dir='/home/pouyas/depth_final',
        transforms=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Train the model
    train_model_with_kfold(
        dataset=train_dataset,
        num_epochs=90,  # Change this according to your needs
        batch_size=2,   # Change this based on your memory availability
        k=5,            # Number of folds
        device=device
    )





    #train_model(model, train_loader, val_loader, device, num_epochs=450)
