from torch import nn

class ResidualUnit(nn.Module):

    ''' Each residual unit for resent groups.'''


    def __init__(self, in_channels:int,
                 mid_channel:int ,
                 out_channels:int, 
                 stride:int=1,dropout_rate:float=0.8):

        '''
        Args:
        in_channels (int): Number of input channels for the residual unit.
        mid_channel (int): Number of channels in the intermediate convolutional layers.
        out_channels (int): Number of output channels for the residual unit.
        stride (int): Stride for the convolutional layers. Default is 1.
        dropout_rate (float): Dropout rate for the dropout layer. Default is 0.8.

        '''
        super(ResidualUnit, self).__init__()

        #block 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channel, kernel_size=1, padding=0)
        #block 2


        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=stride, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
                
        #block 3
        self.bn3 = nn.BatchNorm2d(mid_channel)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0)#shrink feature map with stride

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


    ''' Convolution groups for resunits'''
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

    ''' ResNet architecture for galaxy classification. 
    with modifications to better suit the task and dataset size.'''


    def __init__(self,head:str = "default" ,N=2, k=2, num_classes=5):

        '''
        Args:
            head (str): The type of head to use for the model. Default is "default". 
                        Options include "default", "dropout", "attention", etc.
            N (int): The number of residual units in each group. Default is 2.
            k (int): The width multiplier for the number of channels in the convolutional layers. Default is 2.
            num_classes (int): The number of output classes for classification. Default is 5.
        
        Example:
            ``` python
            model = GalaxyClassifier_ResNet(head="dropout", N=3, k=2, num_classes=5)
            ```


        '''


        super(GalaxyClassifier_ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=2) #128 x 128 -> 64 x 64
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)# downsample to 32 x 32
        #residual entry

        self.group2  = ConvGroup(64,        64*k,  256*k,  N=N, downsample=True)
        self.group3  = ConvGroup(256*k,    128*k,  512*k,  N=N, downsample=True)
        self.group4  = ConvGroup(512*k,    256*k,  1024*k, N=N, downsample=True)
        self.group5  = ConvGroup(1024*k,   512*k,  2048*k, N=N, downsample=False)
        
        #resnet head
        self.avgpool = nn.AdaptiveAvgPool2d(1) # global average pooling to get 1x1 feature map
        self.dropout = nn.Dropout(0.2)
        
        self.fc      = nn.Linear(2048*k, num_classes) #k = 2

        for m in self.modules():# he initializiing to help with convergence as per 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)             # [128, 128] input 
        out = self.maxpool1(out) 
        out = self.group2(out)         
        out = self.group3(out)         
        out = self.group4(out)          
        out = self.group5(out)        
        out = self.avgpool(out)         
        out = out.flatten(1)            
        out = self.dropout(out)
        out = self.fc(out)              
        return out

