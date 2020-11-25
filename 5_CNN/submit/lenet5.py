from torch import nn
import torch.nn.functional as F
# network definition
class lenet5(nn.Module):
    def __init__(self,in_dim,n_class):
        super(lenet5, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1,padding=1
                      ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.BatchNorm2d(6),
            nn.ReLU()
        ) 
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1,padding=1
                      ),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3)
        ) 
        self.linear = nn.Sequential(
            nn.Linear(1920, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.l2(out1)
        # out2 = F.dropout(out2, training=self.training)
        out3 = self.l3(out2)
        # out3 = F.dropout(out3, training=self.training)
        out3 = out3.view(out3.size(0), -1)
        output = self.linear(out3)
        return output