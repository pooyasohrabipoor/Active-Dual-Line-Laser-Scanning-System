import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        return len(self.coco_data['images']) * 1  # Original + Augmented

    def __getitem__(self, idx):
        original_idx = idx // 1
        augmentation_idx = idx % 1

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
        combined_img = np.concatenate([rgb_np, depth_np], axis=2)
        combined_img = Image.fromarray(combined_img.astype(np.uint8))
        combined_np = np.array(combined_img)
   

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


        # Convert images to tensors
        rgb_img = combined_np[:, :, :3]  # RGB channels
        depth_img = combined_np[:, :, 3]  # Depth channel
        rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)
        depth_img = torch.tensor(depth_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension


        # Stack masks to create a [N, H, W] tensor
        target = {
            'boxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks).squeeze(1)
        }

        # Concatenate the depth data as an additional channel to the RGB image
        rgbd_img = torch.cat([rgb_img, depth_img], dim=0)

        return rgbd_img, target

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
    model.transform.image_mean = [0, 0,0,0]  # Add mean for the depth channel
    model.transform.image_std = [1,1,1,1]   # Add std for the depth channel

    return model





val_dataset = CustomDataset(
    annotation_json='/home/pouyas/test/json/test.json',
    rgb_dir='/home/pouyas/test/rgb',
    depth_dir='/home/pouyas/test/depth',
    transforms=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)













model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
model = modify_maskrcnn_for_rgbd(model)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





# Load the trained weights
#load_path = '/home/pouyas/best_RGBD_model_fold_4_epoch_32_iou_0.3829.pth'
load_path ='/home/pouyas/best_RGBD_model_with_RGB_weights_fold_4_epoch_61_iou_0.3831.pth'
model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()  # Set the model to evaluation mode

print(f"Model weights loaded from {load_path}")

# Evaluation function to get AP and AR metrics
def evaluate_model(model, dataset, annotation_file, device, threshold):
    model.eval()

    # Load the COCO-style ground truth annotations
    coco_gt = COCO(annotation_file)  # Ground truth annotations in COCO format
    coco_dt = []

    # Iterate over the dataset and make predictions
    with torch.no_grad():
        print("length of val is",len(dataset))
        for idx in range(len(dataset)):
            rgbd_img, target = dataset[idx]
            if rgbd_img is None:  # Skip invalid images
                continue
            img = rgbd_img.unsqueeze(0).to(device)

            # Make predictions
            outputs = model(img)[0]

            # Prepare predictions in COCO format
            for i, box in enumerate(outputs['boxes']):
                score = outputs['scores'][i].cpu().item()
                if score >= threshold:  # Apply confidence threshold
                    x_min, y_min, x_max, y_max = box.cpu().numpy()
                    width = x_max - x_min
                    height = y_max - y_min
                    coco_dt.append({
                        'image_id': dataset.coco_data['images'][idx]['id'],
                        'category_id': int(outputs['labels'][i].cpu().item()),  # Class label
                        'bbox': [float(x_min), float(y_min), float(width), float(height)],  # Convert bbox format
                        'score': float(score),  # Prediction score
                    })

    # Save predictions to a JSON file (for COCOeval)
    with open('predictions.json', 'w') as f:
        json.dump(coco_dt, f)

    # Load predictions
    coco_dt = coco_gt.loadRes('predictions.json')

    # Evaluate the model
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# Run evaluation on validation dataset
annotation_file = '/home/pouyas/test/json/test.json'  # Ground truth annotations in COCO format
evaluate_model(model, val_dataset, annotation_file, device, threshold=0.9)
