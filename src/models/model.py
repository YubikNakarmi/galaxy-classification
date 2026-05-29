from torch import nn


class ResidualUnit(nn.Module):
    def __init__(self, in_channels,mid_channel ,out_channels, stride=1,dropout_rate=0.5):
        super(ResidualUnit, self).__init__()

        #block 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channel, kernel_size=1, padding=0)
        #block 2
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
                
        #block 3
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channel, out_channels, kernel_size=1,stride=stride, padding=0) #shrink feature map with stride

        '''if the dimensions of the input and output don't match,
        we need to use a convolution to match them for the residual connection'''
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x))) #reverse order than most ResNets since its pre-activation
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.dropout(out)
        out = self.conv3(self.relu3(self.bn3(out)))
        out += identity #residual connection
        return out
    
class ConvGroup(nn.Module):# Using N = 2
    def __init__(self, in_channels, mid_channels, out_channels, N = 2,downsample=True):
        super(ConvGroup, self).__init__()
        units = []
        
        # Unit head, always taking in channels and output out_channels, no downsampling
        units.append(ResidualUnit(in_channels, mid_channels, out_channels, stride=1))
        
        # middle units if N > 2 
        for _ in range(N - 2):
            units.append(ResidualUnit(out_channels, mid_channels, out_channels, stride=1))
        
        # last unit downample
        stride = 2 if downsample else 1
        units.append(ResidualUnit(out_channels, mid_channels, out_channels, stride=stride))
        
        self.units = nn.ModuleList(units)

    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return x
    
class GalaxyClassifier_ResNet(nn.Module):
    def __init__(self, N=2, k=2, num_classes=5):
        super(GalaxyClassifier_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=1, padding=2) #64 x 64 img, 3 channels
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)# downsample to 32 x 32
        #residual entry

      
        self.group2  = ConvGroup(64,   128,  256*k,  N=N, downsample=True)#k = 2
        self.group3  = ConvGroup(512,  256,  1024, N=N, downsample=True)
        self.group4  = ConvGroup(1024, 512,  2048, N=N, downsample=True)
        self.group5  = ConvGroup(2048, 1024, 4096, N=N, downsample=False)

        #resnet head
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(2048*k, num_classes) #k = 2

        for m in self.modules():# he initializiing to help with convergence as per [He et al., 2015](https://arxiv.org/abs/1502.01852)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)             # [B, 64, 64, 64]
        out = self.maxpool1(out)         # [B, 64, 32, 32]
        out = self.group2(out)          # [B, 512, 16, 16]
        out = self.group3(out)          # [B, 1024, 8, 8]
        out = self.group4(out)          # [B, 2048, 4, 4]
        out = self.group5(out)          # [B, 4096, 4, 4]
        out = self.avgpool(out)         # [B, 4096, 1, 1]
        out = out.flatten(1)            # [B, 4096]
        out = self.dropout(out)
        out = self.fc(out)              # [B, 5]
        return out


def model():

    model = GalaxyClassifier_ResNet(N=2, k=2, num_classes=5)