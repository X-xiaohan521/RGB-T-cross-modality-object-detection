import torch
from torch import nn

from src.main.modules import Conv, C3k2, SPPF, C2PSA, Concat, OBB

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
        # x.shape = (batch, 512, 16, 20)

        self.neck2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        # x.shape = (batch, 512, 32, 40)

        self.concat_backbone2_neck2 = Concat(1)
        # x.shape = (batch, 1024, 32, 40)

        self.neck3 = nn.Sequential(C3k2(1024, 512, 2, True, 0.5))
        # x.shape = (batch, 512, 32, 40)

        self.neck4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))
        # x.shape = (batch, 512, 64, 80)

        self.concat_backbone1_neck4 = Concat(1)
        # x.shape = (batch, 1024, 64, 80)

        self.neck5 = nn.Sequential(C3k2(1024, 256, 2, True, 0.5))
        # x.shape = (batch, 256, 64, 80)

        self.neck6 = nn.Sequential(Conv(256, 256, 3, 2, 1))
        # x.shape = (batch, 256, 32, 40)

        self.concat_neck3_neck6 = Concat(1)
        # x.shape = (batch, 768, 32, 40)

        self.neck7 = nn.Sequential(C3k2(768, 512, 2, True, 0.5))
        # x.shape = (batch, 512, 32, 40)

        self.neck8 = nn.Sequential(Conv(512, 512, 3, 2, 1))
        # x.shape = (batch, 512, 16, 20)

        self.concat_neck1_neck8 = Concat(1)
        # x.shape = (batch, 1024, 16, 20)

        self.neck9 = nn.Sequential(C3k2(1024, 1024, 2, True, 0.25))
        # x.shape = (batch, 1024, 16, 20)

        self.detect = OBB(ch=(256, 512, 1024))

    def forward(self, x):
        backbone1_logits = self.backbone1(x)
        # backbone1_logits.shape = (batch, 512, 64, 80)

        backbone2_logits = self.backbone2(backbone1_logits)
        # backbone2_logits.shape = (batch, 512, 32, 40)

        backbone3_logits = self.backbone3(backbone2_logits)
        # backbone2_logits.shape = (batch, 512, 16, 20)

        neck1_logits = self.neck1(backbone3_logits)
        # neck1_logits.shape = (batch, 512, 16, 20)

        neck2_logits = self.neck2(neck1_logits)
        # neck2_logits.shape = (batch, 512, 32, 40)

        concat_backbone2_neck2_logits = self.concat_backbone2_neck2([backbone2_logits, neck2_logits])
        # concat_backbone2_neck2_logits.shape = (batch, 1024, 32, 40)

        neck3_logits = self.neck3(concat_backbone2_neck2_logits)
        # neck3_logits.shape = (batch, 512, 32, 40)

        neck4_logits = self.neck4(neck3_logits)
        # neck4_logits.shape = (batch, 512, 64, 80)

        concat_backbone1_neck4_logits = self.concat_backbone1_neck4([backbone1_logits, neck4_logits])
        # concat_backbone1_neck4_logits.shape = (batch, 1024, 64, 80)

        neck5_logits = self.neck5(concat_backbone1_neck4_logits)  # -> detect
        # neck5_logits.shape = (batch, 256, 64, 80)

        neck6_logits = self.neck6(neck5_logits)
        # neck6_logits.shape = (batch, 256, 32, 40)

        concat_neck3_neck6_logits = self.concat_neck3_neck6([neck3_logits, neck6_logits])
        # concat_neck3_neck6_logits.shape = (batch, 768, 32, 40)

        neck7_logits = self.neck7(concat_neck3_neck6_logits)  # -> detect
        # neck7_logits.shape = (batch, 512, 32, 40)

        neck8_logits = self.neck8(neck7_logits)
        # neck7_logits.shape = (batch, 512, 16, 20)

        concat_neck1_neck8_logits = self.concat_neck1_neck8([neck8_logits, neck1_logits])
        # concat_neck1_neck8_logits.shape = (batch, 1024, 16, 20)

        neck9_logits = self.neck9(concat_neck1_neck8_logits)  # -> detect
        # neck7_logits.shape = (batch, 1024, 16, 20)

        logits = self.detect([
            neck5_logits,   # 256
            neck7_logits,   # 512
            neck9_logits   # 1024
        ])

        return logits

if __name__ == '__main__':
    model = MyOBB()
    print(model)