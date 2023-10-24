import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import _efficientnet_conf, EfficientNet_V2_L_Weights
import warnings
warnings.simplefilter("ignore")
torch.autograd.set_detect_anomaly(True)

inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
lastconv_input_channels = inverted_residual_setting[-1].out_channels
lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
class EfficientFlood(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.backbone = models.efficientnet_v2_l(weights = EfficientNet_V2_L_Weights.DEFAULT)
        del self.backbone.classifier

        self.backbone.fc1d = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(lastconv_output_channels, 512),
            nn.SELU())

        self.backbone.fc2d = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(512, 256),
            nn.SELU())

        self.backbone.fc3d = nn.Sequential(
            nn.Linear(256, num_classes)
            # nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.backbone.features(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.fc1d(x)
        x = self.backbone.fc2d(x)
        x = self.backbone.fc3d(x)

        return x

if __name__ == "__main__":
    image = torch.rand(2,3,331,331)
    model = EfficientFlood(2)
    output = model(image)
    print(output)

    for name, param in model.named_parameters():
        if 'fc3d' not in name and 'fc2d' not in name and 'fc1d' not in name and "features.8." not in name and "features.7" not in name:
            param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)



