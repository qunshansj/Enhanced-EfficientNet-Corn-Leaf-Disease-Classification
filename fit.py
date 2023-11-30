python

# MBConv模块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_factor=6, reduction_ratio=4):
        super(MBConv, self).__init__()

        # 升维
        self.expand = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=1, padding=0)
        self.bn_expand = nn.BatchNorm2d(in_channels * expansion_factor)
        self.relu = nn.ReLU(inplace=True)

        # 深度可分离卷积
        self.depthwise = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                                   kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                                   groups=in_channels * expansion_factor)
        self.bn_depthwise = nn.BatchNorm2d(in_channels * expansion_factor)

        # SE模块
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * expansion_factor, in_channels * expansion_factor // reduction_ratio)
        self.fc2 = nn.Linear(in_channels * expansion_factor // reduction_ratio, in_channels * expansion_factor)

        # 降维
        self.project = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_project = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        # 升维
        x = self.expand(x)
        x = self.bn_expand(x)
        x = self.relu(x)

        # 深度可分离卷积
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu(x)

        # Squeeze Excitation
        b, c, _, _ = x.size()
        se = self.global_avgpool(x).view(b, c)
        se = F.relu(self.fc1(se), inplace=True)
        se = self.fc2(se).sigmoid().view(b, c, 1, 1)
        x = x * se

        # 降维
        x = self.project(x)
        x = self.bn_project(x)

        # 残差连接
        if identity.size() == x.size():
            x += identity
        return x


# CAM模块
class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAM, self).__init__()

        # 全局平均池化和全局最大池化
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)

        # 共享多层感知机
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 平均池化和最大池化得到的特征
        avg_feats = self.global_avgpool(x).view(b, c)
        max_feats = self.global_maxpool(x).view(b, c)

        # 通过共享MLP
        avg_feats = self.fc2(F.relu(self.fc1(avg_feats)))
        max_feats = self.fc2(F.relu(self.fc1(max_feats)))

        # 逐元素的加和操作
        combined = avg_feats + max_feats
        weights = combined.sigmoid().view(b, c, 1, 1)

        return x * weights


# SAM模块
class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        # 空间注意力机制
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1)

    def forward(self, x):
        avg_feats = torch.mean(x, dim=1, keepdim=True)
        max_feats, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_feats, max_feats], dim=1)
        x = self.conv(x).sigmoid()
        return x
......
