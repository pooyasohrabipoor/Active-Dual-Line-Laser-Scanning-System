################################### Maskrcnn RGB  resnet50 ###############
import time 
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.ops import box_iou  # This is the function we will use to compute IoU
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torchvision
import torch
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


import torch
import numpy as np
from PIL import Image
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.ops import box_iou 
# Dataset class
class CustomDataset(Dataset):
    def __init__(self, annotation_json, image_dir, transforms=None):
        with open(annotation_json, 'r') as f:
            self.coco_data = json.load(f)

        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.coco_data['images']) * 5 # Original + Augmented

    def __getitem__(self, idx):
        original_idx = idx // 5
        augmentation_idx = idx % 5

        image_info = self.coco_data['images'][original_idx]
        image_file_name = image_info['file_name']

        # Load RGB image
        rgb_path = os.path.join(self.image_dir, image_file_name)
        rgb_img = Image.open(rgb_path).convert("RGB")

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

            # Ensure mask has shape [1, H, W]
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            masks.append(torch.tensor(mask, dtype=torch.uint8))

        if augmentation_idx > 0 and self.transforms:
            # Apply augmentation (rotation, flip) consistently to images, masks, and bboxes
           # angle = T.RandomRotation.get_params([-60, 60])
           # rgb_img = F.rotate(rgb_img, angle)
           # masks = [F.rotate(mask, angle) for mask in masks]
            #bboxes = self.rotate_bboxes(bboxes, angle, rgb_img.size)

            if random.random() < 0.5:
                rgb_img = F.hflip(rgb_img)
                masks = [F.hflip(mask) for mask in masks]
                bboxes = self.flip_bboxes_horizontal(bboxes, rgb_img.size[0])

            if random.random() < 0.5:
                rgb_img = F.vflip(rgb_img)
                masks = [F.vflip(mask) for mask in masks]
                bboxes = self.flip_bboxes_vertical(bboxes, rgb_img.size[1])

        rgb_img = T.ToTensor()(rgb_img)
        
        # Stack masks to create a [N, H, W] tensor
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks).squeeze(1)
        }

        # Visualization step for both original and augmented data
        #self.visualize(rgb_img, masks, bboxes, augmentation_idx)

        return rgb_img, target

    @staticmethod
    def rotate_bboxes(bboxes, angle, img_size):
        # Update bounding boxes after rotation (this is a basic implementation)
        w, h = img_size
        if angle == 90:
            return [[h - y_max, x_min, h - y_min, x_max] for x_min, y_min, x_max, y_max in bboxes]
        elif angle == 180:
            return [[w - x_max, h - y_max, w - x_min, h - y_min] for x_min, y_min, x_max, y_max in bboxes]
        elif angle == 270:
            return [[y_min, w - x_max, y_max, w - x_min] for x_min, y_min, x_max, y_max in bboxes]
        return bboxes

    @staticmethod
    def flip_bboxes_horizontal(bboxes, img_width):
        # Flip bounding boxes horizontally
        return [[img_width - x_max, y_min, img_width - x_min, y_max] for x_min, y_min, x_max, y_max in bboxes]

    @staticmethod
    def flip_bboxes_vertical(bboxes, img_height):
        # Flip bounding boxes vertically
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
                else:
                    print(f"Unexpected polygon type: {type(polygon)}")
        elif isinstance(segmentation, dict):
            mask = coco_mask.decode(segmentation)
        else:
            raise TypeError(f"Invalid segmentation format: {type(segmentation)}")

        return mask

    import matplotlib.patches as patches

    @staticmethod
    def visualize(rgb_img, masks, bboxes, augmentation_idx):
        rgb_img_np = rgb_img.permute(1, 2, 0).numpy()
        mask_combined = torch.stack(masks).sum(dim=0).squeeze().numpy()

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(rgb_img_np)
        plt.title(f"RGB Image (Augmentation {augmentation_idx})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_combined, cmap='gray')
        plt.title(f"Mask (Augmentation {augmentation_idx})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(rgb_img_np)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='r', facecolor='none'))
        plt.title(f"RGB with Bboxes (Augmentation {augmentation_idx})")
        plt.axis("off")

        plt.show()

# DataLoader setup
def collate_fn(batch):
    return tuple(zip(*batch))

# IoU evaluation
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
                    ious = box_iou(pred_boxes, true_boxes)
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
    train_losses_per_epoch = [] 
    start_time = time.time()

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
        train_losses_per_epoch.append(avg_train_loss)

        # Compute mean IoU on validation set
        current_iou = compute_mean_iou(model, val_loader, device)
        ious_per_epoch.append(current_iou)  # Store IoU
        print(f"Epoch {epoch + 1}, Mean IoU: {current_iou}")

        # Save the model if it achieves the best IoU
        if current_iou > best_iou:
            best_iou = current_iou
            #save_file = f'best_model_epoch_{epoch + 1}_iou_{best_iou:.4f}.pth'
            save_file = 'best_RGB_model_no_k_fold_resnet50.pth'
            torch.save(model.state_dict(), save_file)
            print(f"New best model saved: {save_file}")
    total_time = time.time() - start_time
    print(f"Training completed in: {total_time // 60:.0f} minutes and {total_time % 60:.0f} seconds.")
    # Plot IoU per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ious_per_epoch) + 1), ious_per_epoch, marker='o', label='Mean IoU')
    plt.title('Mean IoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses_per_epoch) + 1), train_losses_per_epoch, marker='o', color='orange', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main function
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
    model.transform.image_mean = [0, 0, 0]  # Add mean for depth channel
    model.transform.image_std = [1, 1, 1]   # Add std for depth channel
    model.to(device)

    # Initialize dataset and dataloader with your paths
    train_dataset = CustomDataset(annotation_json='/home/pouyas/train.json', image_dir='/home/pouyas/rgb_image')
    val_dataset = CustomDataset(annotation_json='/home/pouyas/val.json', image_dir='/home/pouyas/rgb_image')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Train the model
    train_model(model, train_loader, val_loader, device, num_epochs=400)
