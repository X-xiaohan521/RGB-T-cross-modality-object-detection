import torch
from torch import nn

from src.main.modules import Conv, C3k2, SPPF, C2PSA

class MyOBB(nn.Module):
    def __init__(self):
        super(MyOBB, self).__init__()
        # x.shape = (3, 512, 640)
        self.backbone1 = nn.Sequential(Conv(3, 64, 3, 2, 1),
                                       # x.shape = (batch, 64, 256, 320)

                                       Conv(64, 128, 3, 2, 1),
                                       # x.shape = (batch, 128, 128, 160)

                                       C3k2(128, 256, 2, True, 0.25),
                                       # x.shape = (batch, 256, 128, 160)

                                       Conv(256, 256, 3, 2, 1),
                                       # x.shape = (batch, 256, 64, 80)

                                       C3k2(256, 512, 2, True, 0.25)
                                       # x.shape = (batch, 512, 64, 80)
                                   )
        self.backbone2 = nn.Sequential(Conv(512, 512, 3, 2, 1),
                                       # x.shape = (batch, 512, 32, 40)

                                       C3k2(512, 512, 2, True, 0.5)
                                       # x.shape = (batch, 512, 32, 40)
                                   )
        self.backbone3 = nn.Sequential(Conv(512, 512, 3, 2, 1),
                                       # x.shape = (batch, 512, 16, 20)
                                       C3k2(512, 512, 2, True, 0.5)
                                       # x.shape = (batch, 512, 16, 20)
                                   )

        self.neck1 = nn.Sequential(SPPF(512, 512), C2PSA(512, 512))

    def forward(self, x):
        backbone1_logits = self.backbone1(x)
        # backbone1_logits.shape = (batch, 512, 64, 80)

        backbone2_logits = self.backbone2(backbone1_logits)
        # backbone2_logits.shape = (batch, 512, 32, 40)

        backbone3_logits = self.backbone3(backbone2_logits)
        # backbone2_logits.shape = (batch, 512, 16, 20)

        return backbone3_logits

if __name__ == '__main__':
    model = MyOBB()
    print(model)