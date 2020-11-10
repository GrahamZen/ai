class Resnet_cifar(nn.Module):


def __init__(self, n):
    super(Resnet_cifar, self).__init__()
    self.n = n
    self.filter = [16, 32, 64]
    self.bn = nn.BatchNorm2d(self.filter[0])
    self.layer1 = nn.Conv2d(kernel_size=3, in_channels=3, out_chan
                            nels=self.filter[0], padding=1, bias=False, stride=1)
    self.layer2 = self.make_layer(block_num=2*n, inchannel=self.f
                                  ilter[0], channel=self.filter[0], stride=1)
    self.layer3 = self.make_layer(block_num=2*n, inchannel=self.f
                                  ilter[0], channel=self.filter[1], stride=2)
    self.layer4 = self.make_layer(block_num=2*n, inchannel=self.f
                                  ilter[1], channel=self.filter[2], stride=2)
    self.ap = nn.AdaptiveAvgPool2d(1)
    self.linear = nn.Linear(self.filter[2], 10)
    self.apply(_weights_init)


def make_layer(self, block_num, inchannel, channel, stride=1):


layers = []
for idx in range(block_num):
if idx == 0:
layers.append(bottleneck2(inchannel, channel, stride))
else:
layers.append(bottleneck2(channel, channel, 1))
return nn.Sequential(*layers)


def forward(self, x):


output = self.layer1(x)
output = self.bn(output)
output = torch.nn.functional.relu(output)
output = self.layer2(output)
output = self.layer3(output)
output = self.layer4(output)
output = torch.nn.functional.avg_pool2d(output,
                                        output.size()[3])
output = output.view(output.size(0), -1)
output = self.linear(output)
return output
