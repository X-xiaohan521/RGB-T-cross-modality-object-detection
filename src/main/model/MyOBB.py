from collections import OrderedDict

import torch
from torch import nn

from src.main.modules import Conv, C3k2, SPPF, C2PSA, Concat, OBB, v8OBBLoss

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

        self.neck1 = nn.Sequential(SPPF(512, 512), C2PSA(512, 512, 2))
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

        self.neck9 = nn.Sequential(C3k2(1024, 512, 2, True, 0.5))
        # x.shape = (batch, 1024, 16, 20)

        self.detect = OBB(ch=(256, 512, 512))

        # self.loss = v8OBBLoss(self)

    def forward(self, x, label=None):
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

        if label is not None:
            loss = self.loss.loss(logits, x)
            return {
                "logits": logits,
                "loss": loss
            }

        return {"logits": logits}

    def load_backbone_weights(self, pretrained_weight_dict: OrderedDict, verbose: bool = True):
        """
        加载 YOLO v11 L backbone 权重
        """
        if "model" in pretrained_weight_dict:
            pretrained_weight_dict = pretrained_weight_dict["model"].state_dict()
        else:
            pretrained_weight_dict = pretrained_weight_dict

        model_weight_dict = self.state_dict()

        new_state_dict = {}
        unmatched = []
        matched = []

        for k, v in pretrained_weight_dict.items():
            # 映射 backbone 每层名称
            if k.startswith("model.model."):
                k = k.replace("model.model.", "")

            # YOLO backbone → backbone1/2/3
            if k.startswith("0.") or k.startswith("1.") or k.startswith("2.") or k.startswith("3.") or k.startswith(
                    "4."):
                new_k = "backbone1." + k
            elif k.startswith("5.") or k.startswith("6."):
                new_k = "backbone2." + k.replace("5.", "0.").replace("6.", "1.")
            elif k.startswith("7.") or k.startswith("8."):
                new_k = "backbone3." + k.replace("7.", "0.").replace("8.", "1.")
            else:
                continue

            # 检查 shape & 构建新字典
            if new_k in model_weight_dict and model_weight_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                matched.append(new_k)
            else:
                unmatched.append((k, new_k, v.shape))

        # 加载权重
        model_weight_dict.update(new_state_dict)
        self.load_state_dict(model_weight_dict, strict=False)

        if verbose:
            print(f"\nLoaded backbone weights:")
            print(f"✅ Matched layers: {len(matched)}")
            if unmatched:
                print(f"❌ Unmatched layers: {len(unmatched)}")
                print("\n--- Unmatched layers ---")
                for u in unmatched:
                    print(u)

    def load_neck_weights(self, pretrained_weight_dict: OrderedDict, verbose: bool = True):
        """
        加载 YOLO v11 L neck 权重
        """
        if "model" in pretrained_weight_dict:
            pretrained_weight_dict = pretrained_weight_dict["model"].state_dict()
        else:
            pretrained_weight_dict = pretrained_weight_dict

        model_weight_dict = self.state_dict()

        new_state_dict = {}
        unmatched = []
        matched = []

        for k, v in pretrained_weight_dict.items():
            if k.startswith("model.model."):
                k = k.replace("model.model.", "")

            if k.startswith("9."):
                new_k = "neck1." + k.replace("9.", "0.")
            elif k.startswith("10."):
                new_k = "neck1." + k.replace("10.", "1.")
            elif k.startswith("13."):
                new_k = "neck3." + k.replace("13.", "0.")
            elif k.startswith("16."):
                new_k = "neck5." + k.replace("16.", "0.")
            elif k.startswith("17."):
                new_k = "neck6." + k.replace("17.", "0.")
            elif k.startswith("19."):
                new_k = "neck7." + k.replace("19.", "0.")
            elif k.startswith("20."):
                new_k = "neck8." + k.replace("20.", "0.")
            elif k.startswith("22."):
                new_k = "neck9." + k.replace("22.", "0.")
            else:
                continue

            if new_k in model_weight_dict and model_weight_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                matched.append(new_k)
            else:
                unmatched.append((k, new_k, v.shape))

        model_weight_dict.update(new_state_dict)
        self.load_state_dict(model_weight_dict, strict=False)

        if verbose:
            print(f"\nLoaded neck weights:")
            print(f"✅ Matched layers: {len(matched)}")
            if unmatched:
                print(f"❌ Unmatched layers: {len(unmatched)}")
                print("\n--- Unmatched layers ---")
                for u in unmatched:
                    print(u)

if __name__ == '__main__':
    model = MyOBB()
    print(model)