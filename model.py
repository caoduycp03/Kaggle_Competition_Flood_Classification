import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import _efficientnet_conf

model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
lastconv_input_channels = inverted_residual_setting[-1].out_channels
lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(lastconv_output_channels, 2),
    nn.Softmax(dim = 1)
)


if __name__ == "__main__":
    image = torch.rand(2,3,224,224)
    output = model(image)
    print(output)
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    pass