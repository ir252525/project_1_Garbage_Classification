
import torch
import torchvision

from torch import nn
from torchvision.transforms import v2


def creat_effnetb1(num_classes: int= 6,
                   seed : int= 42):
  """Creates an EfficientNetB1 feature extractor model and transforms.

  Args:
      num_classes (int, optional): number of classes in the classifier head.
          Defaults to 3.
      seed (int, optional): random seed value. Defaults to 42.

  Returns:
      model (torch.nn.Module): EffNetB1 feature extractor model.
      transforms (torchvision.transforms): test image transforms.
  """
  transform = v2.Compose([
      v2.Resize(size=(224, 224)),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

  model = torchvision.models.efficientnet_b1(weights="IMAGENET1K_V2")

  for params in model.parameters():
    params.requires_grad = False

  for param in model.features[8].parameters() :
    param.requires_grad = True

  torch.manual_seed(seed)

  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace = True),
      nn.Linear(in_features=1280, out_features=num_classes, bias=True)
      )
  return model, transform
