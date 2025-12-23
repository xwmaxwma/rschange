from torch import nn 
import torch 
from rscd.models.backbones.MambaCSSM import MambaCSSM

class MambaCSSMUnet(nn.Module):

    def __init__(self, output_classes = 2,image_size=256):
        super(MambaCSSMUnet, self).__init__()

        #### Encoder Conv
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.mp_block_1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.mp_block_2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.mp_block_3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.mp_block_4 = nn.MaxPool2d(2, 2, return_indices=True)

        #### Mamba

        feature_size = image_size // 16
        d_model_size = feature_size * feature_size
        self.mamba = MambaCSSM(num_layers=4, d_model=d_model_size, d_conv=4, d_state=16)

        
        #### Decoder Deconv
        self.mpu_block_4 = nn.MaxUnpool2d(2, 2)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, padding=1),
            nn.ReLU()
        )
        self.deconv_4_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_3 = nn.MaxUnpool2d(2, 2)

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU()
        )

        self.deconv_3_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_2 = nn.MaxUnpool2d(2, 2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU()
        )

        self.deconv_2_block = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, 1, padding=1),
            nn.ReLU()
        )

        self.mpu_block_1 = nn.MaxUnpool2d(2, 2)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, padding=1),
            nn.ReLU()
        )

        self.deconv_1_block = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, 3, 1, padding=1),
            nn.ReLU()
        )

        self.conv_final = nn.Conv2d(6, output_classes, 1, 1)


    def forward(self, t1,t2):

        t = torch.cat([t1,t2], dim = 1)

        x1 = self.conv_block_1(t)
        f1, i1 = self.mp_block_1(x1)
        x2 = self.conv_block_2(f1)
        f2, i2 = self.mp_block_2(x2)
        x3 = self.conv_block_3(f2)
        f3, i3 = self.mp_block_3(x3)
        x4 = self.conv_block_4(f3)
        f4, i4 = self.mp_block_4(x4)



        b,c,h,w = f4.shape
        f4_t1 = f4[:,:c//2, :,:]
        f4_t2 = f4[:,c//2:, :,:]



        # print(f4_t1.shape)
        # f4_t1 = f4_t1.view((-1, 64, 16*16))  # Adjusted for input size 256x256
        # f4_t2 = f4_t2.view((-1, 64, 16*16))  # Adjusted for input size 256x256
        # f5_t1,f5_t2 = self.mamba(f4_t1, f4_t2)
        # f5_t1 = f5_t1.view((-1, 64, 16, 16))  # Adjust the shape for further operations
        # f5_t2 = f5_t2.view((-1, 64, 16, 16))  # Adjust the shape for further operations

        b, c, h, w = f4.shape                     # 对 512 输入，此时 h=w=32
        half_c = c // 2                           # 这里仍是 128 -> 一半是 64
        f4_t1 = f4[:, :half_c, :, :]              # [B, 64, h, w]
        f4_t2 = f4[:, half_c:, :, :]              # [B, 64, h, w]

        L = h * w
        f4_t1 = f4_t1.contiguous().view(b, half_c, L)   # [B, 64, h*w]
        f4_t2 = f4_t2.contiguous().view(b, half_c, L)   # [B, 64, h*w]

        f5_t1, f5_t2 = self.mamba(f4_t1, f4_t2)         # 这里的 L 变成 1024

        f5_t1 = f5_t1.contiguous().view(b, half_c, h, w)  # [B, 64, h, w]
        f5_t2 = f5_t2.contiguous().view(b, half_c, h, w)  # [B, 64, h, w]

        f5 = torch.cat([f5_t1, f5_t2], dim = 1)


        f6 = self.mpu_block_4(f5, i4)
        f7 = self.conv_4(torch.cat((x4, f6), dim=1))
        f8 = self.deconv_4_block(f7)

        f9 = self.mpu_block_3(f8, i3, output_size=x3.size())
        f10 = self.conv_3(torch.cat((f9, x3), dim=1))
        f11 = self.deconv_3_block(f10)

        f12 = self.mpu_block_2(f11, i2)
        f13 = self.conv_2(torch.cat((f12, x2), dim=1))

        f14 = self.deconv_2_block(f13)

        f15 = self.mpu_block_1(f14, i1)
        f16 = self.conv_1(torch.cat((f15, x1), dim=1))
        f17 = self.deconv_1_block(f16)
        f18 = self.conv_final(f17)






        return f18