!pip install dtcwt
!pip install grad-cam
!pip install opencv-python

import numpy as np
import torch
import torchvision
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp  # For mixed precision training
import dtcwt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
from torchvision.transforms.functional import rotate, hflip, vflip, adjust_brightness, adjust_contrast
from collections import Counter
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

# WaveletTransform
class WaveletTransform:
    def __init__(self, level=1):
        self.transform = dtcwt.Transform2d()  # Using Dual-Tree Complex Wavelet Transform
        self.level = level

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32)  # Ensure type compatibility
        if img_array.ndim == 3:
            transformed = [self.transform.forward(img_array[:, :, i], nlevels=self.level) for i in range(img_array.shape[2])]
            coeffs = [x.lowpass for x in transformed]  # Extract lowpass components
            img_array = np.stack(coeffs, axis=2)
        img = torchvision.transforms.functional.to_pil_image(img_array.astype(np.uint8))
        return img

#transformations for dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    WaveletTransform(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('/kaggle/input/kindey-stone-dataset-splitted/Kindey_Stone_Dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('/kaggle/input/kindey-stone-dataset-splitted/Kindey_Stone_Dataset/val', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

val_dataset.__getitem__(5)[0].shape

image = val_dataset.__getitem__(5)[0].permute(1, 2, 0).numpy()

# Display the image
plt.imshow(image)
plt.axis('off')  # Hide axis
plt.show()

# Number of classes
classes = train_dataset.classes
num_classes = len(classes)
print("Classes:", classes)

def channel_shuffle(x, groups=2):
  bat_size, channels, w, h = x.shape
  group_c = channels // groups
  x = x.view(bat_size, groups, group_c, w, h)
  x = t.transpose(x, 1, 2).contiguous()
  x = x.view(bat_size, -1, w, h)
  return x

# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )

def conv_bn(in_c, out_c, stride=2):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
    nn.BatchNorm2d(out_c),
    nn.ReLU(True)
  )


class ShuffleBlock(nn.Module):
  def __init__(self, in_c, out_c, downsample=False):
    super(ShuffleBlock, self).__init__()
    self.downsample = downsample
    half_c = out_c // 2
    if downsample:
      self.branch1 = nn.Sequential(
          # 3*3 dw conv, stride = 2
          nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
          nn.BatchNorm2d(in_c),
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )

      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 2
          nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )
    else:
      # in_c = out_c
      assert in_c == out_c

      self.branch2 = nn.Sequential(
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True),
          # 3*3 dw conv, stride = 1
          nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
          nn.BatchNorm2d(half_c),
          # 1*1 pw conv
          nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
          nn.BatchNorm2d(half_c),
          nn.ReLU(True)
      )


  def forward(self, x):
    out = None
    if self.downsample:
      # if it is downsampling, we don't need to do channel split
      out = t.cat((self.branch1(x), self.branch2(x)), 1)
    else:
      # channel split
      channels = x.shape[1]
      c = channels // 2
      x1 = x[:, :c, :, :]
      x2 = x[:, c:, :, :]
      out = t.cat((x1, self.branch2(x2)), 1)
    return channel_shuffle(out, 2)


class ShuffleNet2(nn.Module):
  def __init__(self, num_classes=4, input_size=val_dataset.__getitem__(5)[0].shape[-1], net_type=0.5):
    super(ShuffleNet2, self).__init__()
    assert input_size % 32 == 0


    self.stage_repeat_num = [4, 8, 4]
    if net_type == 0.5:
      self.out_channels = [3, 24//4, 48//2, 96//4, 192//4, 1024//16]
    elif net_type == 1:
      self.out_channels = [3, 24//4, 116//2, 232//4, 464//8, 1024//16]
    elif net_type == 1.5:
      self.out_channels = [3, 24//4, 176//4, 352//4, 704//4, 1024//4]
    elif net_type == 2:
      self.out_channels = [3, 24//4, 244//4, 488//4, 976//4, 2948//4]
    else:
      print("the type is error, you should choose 0.5, 1, 1.5 or 2")

    # let's start building layers
    self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    in_c = self.out_channels[1]

    self.stages = []
    for stage_idx in range(len(self.stage_repeat_num)):
      out_c = self.out_channels[2+stage_idx]
      repeat_num = self.stage_repeat_num[stage_idx]
      for i in range(repeat_num):
        if i == 0:
          self.stages.append(ShuffleBlock(in_c, out_c, downsample=True))
        else:
          self.stages.append(ShuffleBlock(in_c, in_c, downsample=False))
        in_c = out_c
    self.stages = nn.Sequential(*self.stages)

    in_c = self.out_channels[-2]
    out_c = self.out_channels[-1]
    self.conv5 = conv_1x1_bn(in_c, out_c, 1)
    self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(16))

    # fc layer
    self.fc = nn.Linear(out_c, num_classes)


  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.stages(x)
    x = self.conv5(x)
    x = self.g_avg_pool(x)
    x = x.view(-1, self.out_channels[-1])
    x = self.fc(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShuffleNet2().to(device)#models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = amp.GradScaler()  # For mixed precision

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

def train_model(model, criterion, optimizer, train_loader,scheduler, val_loader, epochs=40):
    model.train()
    best_val_accuracy = 0.0
    best_model_wts = None

    for epoch in range(epochs):
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        print('train : ', epoch)

        model.train()  # Ensure model is in training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with amp.autocast():  # Mixed precision context
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item() * inputs.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                total_val_loss += loss.item() * inputs.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        # Step the scheduler based on validation accuracy
        scheduler.step(val_accuracy)

        # Check if this is the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()

    print('Training complete')

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'kidney-stone-shufflenetv2c-dtcwt-1.pth')
        print(f'Best model saved with validation accuracy: {best_val_accuracy:.2f}%')

train_model(model, criterion, optimizer, train_loader, scheduler, val_loader)

# Load best model for evaluation
model.load_state_dict(torch.load('kidney-stone-shufflenetv2c-dtcwt-1.pth'))
model.eval()

# Evaluate on validation set
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# classification report and confusion matrix
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.manifold import TSNE
def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, labels_batch in dataloader:
            inputs = inputs.to(device)
            labels.extend(labels_batch.numpy())

            # Forward pass through ShuffleNetV2 layers
            x = model.conv1(inputs)
            x = model.maxpool(x)
            x = model.stages(x)
            x = model.conv5(x)
            x = model.g_avg_pool(x)
            x = x.view(x.size(0), -1)  # Flatten the features

            features.append(x.cpu().numpy())
    features = np.vstack(features)
    return features, np.array(labels)
# Extract features and labels
features, labels = extract_features(model, val_loader)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Perform t-SNE with specified parameters
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
tsne_results = tsne.fit_transform(features)

# Plot t-SNE results using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette='viridis', legend='full')

plt.title('t-SNE projection of the features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Clean up existing hooks
        if hasattr(self.target_layer, '_forward_hooks'):
            self.target_layer._forward_hooks.clear()

        if hasattr(self.target_layer, '_backward_hooks'):
            self.target_layer._backward_hooks.clear()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        target = torch.zeros_like(output)
        target[:, target_class] = 1
        output.backward(gradient=target)

        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        grad_cam_map = (weights * activations).sum(1, keepdim=True)

        grad_cam_map = nn.functional.relu(grad_cam_map)
        grad_cam_map = nn.functional.interpolate(grad_cam_map, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        grad_cam_map = grad_cam_map - grad_cam_map.min()
        grad_cam_map = grad_cam_map / grad_cam_map.max()

        return grad_cam_map

def visualize_grad_cam(images, model, target_layer, num_images=5):
    model.eval()
    grad_cam = GradCAM(model, target_layer)

    grad_cam_maps = []
    for i in range(num_images):
        input_image = images[i].unsqueeze(0).to(device)
        grad_cam_map = grad_cam.generate_cam(input_image)
        grad_cam_maps.append(grad_cam_map)

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axs[i]
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        ax.imshow(grad_cam_maps[i].squeeze().detach().cpu().numpy(), cmap='jet', alpha=0.5)
        ax.axis('off')
    plt.show()

# validation batch
images, labels = next(iter(val_loader))

# correct layer from ShuffleNet2
target_layer = model.conv5

visualize_grad_cam(images, model, target_layer, num_images=4)

