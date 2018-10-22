import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    #如果不用这个，则需要手动计算卷积网络的输出；
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    #所有的向前传递的逻辑都必须实现在这个方法里面，默认的实现就是放入一个nn。这里实现了两个nn的连接
    def forward(self, x):
        #1、先将输入参数传入到卷积nn，将得到的结果变成一维；
        #2、将一维的输出作为输入，传入到全联网络；
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
