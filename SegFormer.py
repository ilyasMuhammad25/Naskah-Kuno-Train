import os
import numpy as np
import random
from glob import glob
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import wandb

# -----------------------------
# Config & Reproducibility
# -----------------------------
CONFIG = {
    "image_size": 512,
    "batch_size": 8,
    "val_batch_size": 4,
    "epochs": 100,  # Lebih banyak epoch dengan early stopping
    "lr": 6e-5,
    "min_lr": 1e-7,
    "warmup_epochs": 5,
    "num_classes": 5,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
    "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",  # B2 lebih baik dari B0
    "gradient_accumulation_steps": 2,  # Effective batch size = 16
    "mixed_precision": True,  # AMP untuk efisiensi memory
    "patience": 15,  # Early stopping patience
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "focal_loss_gamma": 2.0,  # Untuk mengatasi class imbalance
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

# -----------------------------
# Custom Loss Functions
# -----------------------------
class FocalLoss(nn.Module):
    """Focal Loss untuk mengatasi class imbalance"""
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss.view(-1)
            focal_loss = focal_loss.view(targets.shape)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss untuk segmentasi"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        targets_one_hot = nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        inputs_softmax = nn.functional.softmax(inputs, dim=1)
        
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(2, 3))
        union = inputs_softmax.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """Kombinasi Focal Loss + Dice Loss"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

# -----------------------------
# Enhanced Dataset with Augmentation
# -----------------------------
class FungiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_processor, is_train=True, use_augmentation=True):
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.image_paths = []
        self.image_processor = image_processor
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train

        valid_pairs = []
        for mask_path in self.mask_paths:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            if os.path.exists(image_path):
                valid_pairs.append((image_path, mask_path))

        if not valid_pairs:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

        self.image_paths, self.mask_paths = zip(*valid_pairs)
        
        # Data augmentation pipeline
        if self.use_augmentation:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=30, 
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.MotionBlur(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.CLAHE(p=0.5),
                    A.HueSaturationValue(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.5),
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=32, 
                    max_width=32, 
                    p=0.3
                ),
            ])
        
        print(f"✅ Loaded {len(self.image_paths)} image-mask pairs for {'training' if is_train else 'validation'}")
        if self.use_augmentation:
            print("   📊 Data augmentation enabled")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Convert to numpy for augmentation
        if self.use_augmentation:
            image_np = np.array(image)
            mask_np = np.array(mask)
            
            # Apply augmentations
            augmented = self.transform(image=image_np, mask=mask_np)
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Preprocessing menggunakan image_processor
        inputs = self.image_processor(images=image, segmentation_maps=mask, return_tensors="pt")

        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = inputs["labels"].squeeze(0).long()
        
        # Clamp labels untuk keamanan
        labels = torch.clamp(labels, 0, CONFIG["num_classes"] - 1)
        
        return pixel_values, labels

# -----------------------------
# Enhanced Metrics with Per-Class IoU and Confusion Matrix
# -----------------------------
def calculate_metrics(pred, target, num_classes):
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    precision = precision_score(target_flat, pred_flat, average='macro', zero_division=0)
    recall = recall_score(target_flat, pred_flat, average='macro', zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='macro', zero_division=0)
    acc = accuracy_score(target_flat, pred_flat)

    # Confusion Matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=list(range(num_classes)))
    
    # Per-class IoU
    iou_scores = []
    class_names = ['background', 'class1', 'class2', 'class3', 'class4']  # Sesuaikan dengan dataset
    per_class_iou = {}
    
    pred_cpu = pred.view(-1).cpu()
    target_cpu = target.view(-1).cpu()
    
    for cls in range(num_classes):
        pred_inds = (pred_cpu == cls)
        target_inds = (target_cpu == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        iou_scores.append(iou)
        per_class_iou[f'IoU_{class_names[cls]}'] = iou

    miou = np.nanmean(iou_scores)
    
    return precision, recall, f1, acc, miou, per_class_iou, cm

def plot_confusion_matrix(cm, class_names, epoch, phase='val', normalize=False, save_path=None):
    """
    Plot confusion matrix dengan seaborn
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        epoch: Current epoch
        phase: 'train' or 'val'
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Unnormalized confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, ax=ax1, square=True)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_title(f'Confusion Matrix - {phase.capitalize()} (Epoch {epoch})', fontsize=14)
    
    # Normalized confusion matrix (percentage)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, ax=ax2, square=True, vmin=0, vmax=100)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_title(f'Normalized Confusion Matrix (%) - {phase.capitalize()} (Epoch {epoch})', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

def calculate_per_class_metrics_from_cm(cm, class_names):
    """
    Calculate per-class metrics from confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    per_class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'support': cm[i, :].sum()  # Number of true instances
        }
    
    return per_class_metrics

# -----------------------------
# Training with Mixed Precision and Gradient Accumulation
# -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, num_classes, 
                   gradient_accumulation_steps, use_amp=True, custom_loss=None):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    optimizer.zero_grad()
    
    for i, (pixel_values, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Mixed precision training
        with autocast(enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            
            if custom_loss is not None:
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], 
                    mode="bilinear", align_corners=False
                )
                loss = custom_loss(upsampled_logits, labels)
            else:
                loss = outputs.loss
            
            loss = loss / gradient_accumulation_steps

        # Backpropagation with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (i + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps

        # Predictions
        with torch.no_grad():
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], 
                mode="bilinear", align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    prec, rec, f1, acc, miou, per_class_iou, cm = calculate_metrics(all_preds, all_targets, num_classes)

    return total_loss / len(loader), prec, rec, f1, acc, miou, per_class_iou, cm

def validate(model, loader, device, num_classes, custom_loss=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for pixel_values, labels in tqdm(loader, desc="Validating", leave=False):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            
            if custom_loss is not None:
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], 
                    mode="bilinear", align_corners=False
                )
                loss = custom_loss(upsampled_logits, labels)
            else:
                loss = outputs.loss

            total_loss += loss.item()

            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], 
                mode="bilinear", align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    prec, rec, f1, acc, miou, per_class_iou, cm = calculate_metrics(all_preds, all_targets, num_classes)

    return total_loss / len(loader), prec, rec, f1, acc, miou, per_class_iou, cm

# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score <= self.best_score + self.min_delta) or \
             (self.mode == 'min' and score >= self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    wandb.init(
        project="segmentasi-jamurnet",
        name="segformer-b2-enhanced",
        config=CONFIG
    )
    
    # Class names - SESUAIKAN dengan dataset Anda
    class_names = ['background', 'class1', 'class2', 'class3', 'class4']
    
    # Create directories for saving confusion matrices
    os.makedirs('confusion_matrices', exist_ok=True)
    
    # Initialize image processor
    image_processor = SegformerImageProcessor.from_pretrained(
        CONFIG["model_name"],
        do_reduce_labels=False
    )

    # Create datasets
    train_dataset = FungiDataset(
        "./dataset_masks/train/images",
        "./dataset_masks/train/label_masks",
        image_processor,
        is_train=True,
        use_augmentation=True
    )
    val_dataset = FungiDataset(
        "./dataset_masks/valid/images",
        "./dataset_masks/valid/label_masks",
        image_processor,
        is_train=False,
        use_augmentation=False
    )

    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["val_batch_size"], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_classes"],
        ignore_mismatched_sizes=True,
    ).to(CONFIG["device"])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=CONFIG["min_lr"]
    )
    
    # Initialize custom loss
    custom_loss = CombinedLoss(
        focal_weight=0.6, 
        dice_weight=0.4,
        focal_gamma=CONFIG["focal_loss_gamma"]
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if CONFIG["mixed_precision"] else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG["patience"], mode='max')
    
    best_val_miou = -1.0
    best_val_f1 = -1.0
    best_cm = None

    for epoch in range(CONFIG["epochs"]):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'='*50}")

        # Training
        train_loss, train_prec, train_rec, train_f1, train_acc, train_miou, train_per_class, train_cm = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            CONFIG["device"], CONFIG["num_classes"],
            CONFIG["gradient_accumulation_steps"],
            CONFIG["mixed_precision"],
            custom_loss
        )

        # Validation
        val_loss, val_prec, val_rec, val_f1, val_acc, val_miou, val_per_class, val_cm = validate(
            model, val_loader, CONFIG["device"], CONFIG["num_classes"], custom_loss
        )
        
        # Calculate per-class metrics from confusion matrix
        train_class_metrics = calculate_per_class_metrics_from_cm(train_cm, class_names)
        val_class_metrics = calculate_per_class_metrics_from_cm(val_cm, class_names)

        # Create confusion matrix plots
        # Save confusion matrix every 5 epochs or if it's the best model
        if (epoch + 1) % 5 == 0 or val_miou > best_val_miou:
            # Training confusion matrix
            train_cm_fig = plot_confusion_matrix(
                train_cm, class_names, epoch + 1, 'train',
                save_path=f'confusion_matrices/train_cm_epoch_{epoch+1}.png'
            )
            
            # Validation confusion matrix
            val_cm_fig = plot_confusion_matrix(
                val_cm, class_names, epoch + 1, 'val',
                save_path=f'confusion_matrices/val_cm_epoch_{epoch+1}.png'
            )
            
            # Log confusion matrices to wandb
            wandb.log({
                "train_confusion_matrix": wandb.Image(train_cm_fig),
                "val_confusion_matrix": wandb.Image(val_cm_fig),
            })
            
            plt.close(train_cm_fig)
            plt.close(val_cm_fig)

        # Logging to wandb
        log_dict = {
            "epoch": epoch + 1,
            "learning_rate": scheduler.get_last_lr()[0],
            "train/loss": train_loss,
            "train/precision": train_prec,
            "train/recall": train_rec,
            "train/f1": train_f1,
            "train/accuracy": train_acc,
            "train/mIoU": train_miou,
            "val/loss": val_loss,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/f1": val_f1,
            "val/accuracy": val_acc,
            "val/mIoU": val_miou,
        }
        
        # Add per-class IoU to log
        for cls_name, iou_val in val_per_class.items():
            if not np.isnan(iou_val):
                log_dict[f"val/{cls_name}"] = iou_val
        
        # Add per-class metrics from confusion matrix
        for cls_name, metrics in val_class_metrics.items():
            log_dict[f"val/{cls_name}_precision_cm"] = metrics['precision']
            log_dict[f"val/{cls_name}_recall_cm"] = metrics['recall']
            log_dict[f"val/{cls_name}_f1_cm"] = metrics['f1']
            log_dict[f"val/{cls_name}_specificity"] = metrics['specificity']
        
        # Log confusion matrix as table
        wandb.log({
            "val_confusion_matrix_table": wandb.Table(
                columns=["True\Pred"] + class_names,
                data=[[class_names[i]] + val_cm[i].tolist() for i in range(len(class_names))]
            )
        })
        
        wandb.log(log_dict)

        # Print metrics
        print(f"📈 Train - Loss: {train_loss:.4f} | F1: {train_f1:.4f} | mIoU: {train_miou:.4f}")
        print(f"📊 Val   - Loss: {val_loss:.4f} | F1: {val_f1:.4f} | mIoU: {val_miou:.4f}")
        
        # Print per-class performance
        print(f"\n📊 Per-Class Validation Metrics (from Confusion Matrix):")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Specificity':<10} {'Support':<10}")
        print("-" * 65)
        for cls_name, metrics in val_class_metrics.items():
            print(f"{cls_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {metrics['specificity']:<10.3f} {int(metrics['support']):<10}")

        # Save best model based on mIoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_f1 = val_f1
            best_cm = val_cm
            
            # Save full model
            save_path = "segformer_fungi_best_model_full.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_miou': best_val_miou,
                'best_val_f1': best_val_f1,
                'confusion_matrix': best_cm,
                'class_names': class_names,
                'config': CONFIG
            }, save_path)
            
            # Save best confusion matrix separately
            best_cm_fig = plot_confusion_matrix(
                best_cm, class_names, epoch + 1, 'best_val',
                save_path='confusion_matrices/best_val_cm.png'
            )
            plt.close(best_cm_fig)
            
            print(f"✅ Model saved! New best validation mIoU: {best_val_miou:.4f}")
            
            # Log best metrics
            wandb.run.summary["best_val_miou"] = best_val_miou
            wandb.run.summary["best_val_f1"] = best_val_f1
            wandb.run.summary["best_epoch"] = epoch + 1

        # Early stopping check
        if early_stopping(val_miou):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break

    # Final confusion matrix analysis
    if best_cm is not None:
        print(f"\n{'='*50}")
        print("📊 BEST MODEL CONFUSION MATRIX ANALYSIS")
        print(f"{'='*50}")
        
        best_class_metrics = calculate_per_class_metrics_from_cm(best_cm, class_names)
        print(f"\n{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Specificity':<10}")
        print("-" * 55)
        for cls_name, metrics in best_class_metrics.items():
            print(f"{cls_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {metrics['specificity']:<10.3f}")

    print(f"\n{'='*50}")
    print("✅ Training complete!")
    print(f"📊 Best Val mIoU: {best_val_miou:.4f} | Best Val F1: {best_val_f1:.4f}")
    wandb.finish()

# -----------------------------
# Inference Function untuk Testing
# -----------------------------
def test_model_with_confusion_matrix(model_path, test_loader, device, num_classes, class_names):
    """
    Test model dan generate confusion matrix untuk test set
    
    Args:
        model_path: Path to saved model
        test_loader: DataLoader untuk test set
        device: Device untuk inference
        num_classes: Number of classes
        class_names: List of class names
    """
    print("\n🔍 Running model testing with confusion matrix...")
    
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        CONFIG["model_name"],
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for pixel_values, labels in tqdm(test_loader, desc="Testing"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], 
                mode="bilinear", align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    prec, rec, f1, acc, miou, per_class_iou, cm = calculate_metrics(all_preds, all_targets, num_classes)
    
    # Print results
    print(f"\n📊 TEST SET RESULTS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"mIoU: {miou:.4f}")
    
    # Per-class metrics
    class_metrics = calculate_per_class_metrics_from_cm(cm, class_names)
    print(f"\n📊 Per-Class Test Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10}")
    print("-" * 55)
    
    for i, cls_name in enumerate(class_names):
        metrics = class_metrics[cls_name]
        iou_value = per_class_iou.get(f'IoU_{cls_name}', 0)
        print(f"{cls_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1']:<10.3f} {iou_value:<10.3f}")
    
    # Save test confusion matrix
    test_cm_fig = plot_confusion_matrix(
        cm, class_names, 0, 'test',
        save_path='confusion_matrices/test_confusion_matrix.png'
    )
    plt.close(test_cm_fig)
    
    print(f"\n✅ Test confusion matrix saved to 'confusion_matrices/test_confusion_matrix.png'")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'miou': miou,
        'confusion_matrix': cm,
        'per_class_metrics': class_metrics
    }

if __name__ == "__main__":
    main()
