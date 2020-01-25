import torch.nn as nn
import torch.nn.init as init


class feature_extractor(nn.Module):

    def __init__(self):

        super(feature_extractor,self).__init__()

        self.main = nn.Sequential(
            nn.Sequential(

                nn.Conv2d(3, 20, 5,stride=1),
                nn.MaxPool2d(2, stride=2, dilation=1),
                nn.ReLU(True),

                nn.Conv2d(20, 50, 5, stride=1),
                nn.Dropout2d(False),
                nn.MaxPool2d(2, stride=2, dilation=1),
                nn.ReLU(True)
            )
        )
        self.fc = nn.Linear(800,500)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        x = self.main(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc(x)

        return x

class class_classifier(nn.Module):
    def __init__(self):
        super(class_classifier, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(500,10)
        )

    def forward(self, x):
        x = self.main(x)

        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(500,500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(500, 2),
        )

    def forward(self, x):
        x = self.main(x)

        return x