import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config["num_classes"]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class Resnet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


class Resnet152(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


class Resnext101(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


class Inceptionv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


class Mobilenetv3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Linear(1280, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.classifier.weight)

    def forward(self, x):
        return self.model(x)


class Densenet121(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.classifier.weight)

    def forward(self, x):
        return self.model(x)


class Densenet161(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(2208, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.classifier.weight)

    def forward(self, x):
        return self.model(x)


class Densenet201(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.model = models.densenet201(pretrained=True)
        self.model.classifier = nn.Linear(1920, self.num_classes, bias=True)
        nn.init.kaiming_normal_(self.model.classifier.weight)

    def forward(self, x):
        return self.model(x)