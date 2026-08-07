import torch
import torch.nn as nn
from torchvision.models import densenet121


class DenseNet_121(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNet_121, self).__init__()
        model = densenet121()
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier 교체 (num_classes 맞춤)
        in_features = model.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature_list = []

        # DenseNet의 특징 추출 부분
        x = self.features(x)
        feature_list.append(x)

        # pool + flatten
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)

        # 최종 분류기
        logits_cls = self.classifier(feature)

        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold, return_feature=False):
        x = self.features(x)
        x = self.avgpool(x)
        feature = x.clip(max=threshold)
        feature = torch.flatten(feature, 1)
        logits_cls = self.classifier(feature)
        if return_feature:
            return logits_cls, feature
        return logits_cls

    def get_fc(self):
        return (
            self.classifier.weight.cpu().detach().numpy(),
            self.classifier.bias.cpu().detach().numpy(),
        )

    def get_fc_layer(self):
        return self.classifier
    
    def get_W(self):
        return self.classifier.weight