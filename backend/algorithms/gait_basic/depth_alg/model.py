import torch.nn as nn


class SignalNet(nn.Module): # [85, 129] -> [N]

    def __init__(self, in_channels=85, num_of_class=6):
        super(SignalNet, self).__init__()

        if in_channels <= 64:
            middle_channels = 64
        else:
            middle_channels = 128
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(middle_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
        )

        self.linear = nn.Sequential(
            nn.Linear(4096 ,1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, num_of_class),
        )

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
