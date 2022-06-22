import torch.nn as nn
import torch
import torch.nn.functional as F

from util import Trapz_Integration,mul_Conv2d
from scipy import integrate
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class myNet(nn.Module):
    def __init__(self,init_weights=False):
        super(myNet, self).__init__()

        self.en_and_de1=En_Decoder(16,1,kernel_size=(1,5),padding=(0,2),n=1)
        self.en_and_de2=En_Decoder(16,1,kernel_size=(1,5),padding=(0,2),n=1)
        # self.en_and_de1_1 = En_Decoder(1, 1, kernel_size=(1, 3),padding=(0,1), n=1)
        # self.avg_pool_st=nn.AvgPool2d(kernel_size=(1,3),padding=(0,1),stride=(1,1))

        self.conv_a1=BasicConv2d(1,16,kernel_size=(1,5),stride=(1,1),padding=(0,2))
        self.conv_s1=BasicConv2d(1,16,kernel_size=(1,7),stride=(1,1),padding=(0,3))
        self.conv_a2 = BasicConv2d(16, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.conv_s2 = BasicConv2d(16, 16, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        # self.avg_s1=nn.AvgPool2d(kernel_size=(1,7),stride=(1,1),padding=(0,3))
        # self.avg_s2 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        # self.conv1=BasicConv2d(1,64,kernel_size=(2,5),stride=(1,1),padding=(0,2))
        # self.en_and_de_a=En_Decoder(64,1,kernel_size=(1,3),n=1)
        # self.conv_add_1=BasicConv2d(1,16,kernel_size=(2,5),stride=(1,1),padding=(0,2))
        # self.conv_add_2=BasicConv2d(16,16,kernel_size=(1,5),stride=(1,1),padding=(0,2))
        # self.conv_add_3 = BasicConv2d(16, 8, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        # self.conv_add_4 = nn.Conv2d(8, 1, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))

        # self.en_and_de3=En_Decoder2(1,1,kernel_size=(1,3),padding=(0,1),n=1)
        self.conv_A_3=BasicConv2d(1,16,kernel_size=(1,3),stride=(1,1),padding=(0,1))
        self.conv_B_3=BasicConv2d(16,16,kernel_size=(1,3),stride=(1,1),padding=(0,1))
        self.en_and_de3 = En_Decoder(16, 1, kernel_size=(1, 3), padding=(0, 1), n=1)
        self.dropout1=nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)


        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,0.5,nonlinearity='leaky_relu')
                # nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x0):
        x_a=x0[:,:,0:1,:]
        x_s=x0[:,:,1:2,:]
        x_a=self.conv_a1(x_a)
        x_s=self.conv_s1(x_s)
        # x_a = self.dropout1(x_a)
        # x_s=self.dropout2(x_s)
        x_a=self.conv_a2(x_a)
        x_s=self.conv_s2(x_s)
        # x_a=self.dropout2(x_a)
        # x_s=self.dropout3(x_s)
        # x_a=self.en_and_de1(x_a)
        dis_hi=self.en_and_de1(x_a)
        dis_pseudo=self.en_and_de2(x_s)


        displacement=self.en_and_de3(self.conv_B_3(self.conv_A_3(dis_hi+dis_pseudo)))
        # displacement = self.en_and_de_a(x0)
        return dis_hi,dis_pseudo,displacement


class En_Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,n=1):
        super(En_Decoder, self).__init__()
        self.conv_A=BasicConv2d(in_channels=in_channels,out_channels=int(64*n),kernel_size=kernel_size,stride=(1,2),padding=padding)

        self.conv_B = BasicConv2d(in_channels=int(64*n), out_channels=int(64*n), kernel_size=kernel_size, stride=(1, 2),
                                  padding=padding)
        self.conv_C = BasicConv2d(in_channels=int(64*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 2),
                                  padding=padding)
        # self.conv_C_1=BasicConv2d(in_channels=int(64*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 1),
        #                           padding=padding)
        self.conv_D = BasicConv2d(in_channels=int(32*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 2),
                                  padding=padding)
        # # self.conv_D1=BasicConv2d(in_channels=int(32*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 2),
        #                           padding=padding)
        # self.conv_D_1=BasicConv2d(in_channels=int(32*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 1),
        #                           padding=padding)
        self.deconv_A=BasicTConv2d(in_channels=int(32*n),out_channels=int(32*n),kernel_size=(1,4),stride=(1,2),padding=(0,1))
        # # self.deconv_A1 = BasicTConv2d(in_channels=int(32 * n), out_channels=int(32 * n), kernel_size=(1, 2),
        #                              stride=(1, 2), padding=(0, 0))
        # self.conv_de_A_1=BasicConv2d(in_channels=int(32*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 1),
        #                           padding=padding)
        self.deconv_B = BasicTConv2d(in_channels=int(32*n), out_channels=int(64*n), kernel_size=(1,4),stride=(1,2),padding=(0,1))
        # self.conv_de_B_1=BasicConv2d(in_channels=int(64*n), out_channels=int(64*n), kernel_size=kernel_size, stride=(1, 1),
        #                           padding=padding)
        self.deconv_C = BasicTConv2d(in_channels=int(64*n), out_channels=int(64*n), kernel_size=(1, 4), stride=(1, 2),padding=(0,1))
        self.deconv_D = BasicTConv2d(in_channels=int(64*n), out_channels=int(16*n), kernel_size=(1, 4), stride=(1, 2),padding=(0,1))
        self.conv_E=nn.Conv2d(in_channels=int(16*n),out_channels=out_channels,kernel_size=kernel_size,stride=(1,1),padding=padding)

        # self.dropout_a1=nn.Dropout(0.1)
        # self.dropout_a2=nn.Dropout(0.1)
        # self.dropout_a3 = nn.Dropout(0.05)
        # self.dropout_a4 = nn.Dropout(0.05)
        # self.dropout_a5 = nn.Dropout(0.05)

    def forward(self,x):
        xa=self.conv_A(x)
        # xa=self.dropout_a1(xa)
        xb=self.conv_B(xa)
        # xb=self.dropout_a2(xb)
        xc=self.conv_C(xb)
        # xc=self.dropout_a3(xc)
        # xc_1=self.conv_C_1(xc)
        xd=self.conv_D(xc)
        # xd=self.dropout_a4(xd)
        # xd=self.conv_D_1(xd)
        # xd=self.dropout_a5(xd)
        # xd_1=self.conv_D1(xd)
        xd=self.deconv_A(xd)
        # x_f=self.deconv_A1(xd_1+xd)
        x_f=self.deconv_B(xd+xc)
        x_f=self.deconv_C(x_f+xb)
        x_f=self.deconv_D(x_f+xa)
        x_f=self.conv_E(x_f+x)
        x=x_f
        return x




class En_Decoder2(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,n=1):
        super(En_Decoder2, self).__init__()
        self.conv_A=BasicConv2d(in_channels=in_channels,out_channels=int(32*n),kernel_size=kernel_size,stride=(1,2),padding=padding)

        self.conv_B = BasicConv2d(in_channels=int(32*n), out_channels=int(32*n), kernel_size=kernel_size, stride=(1, 2),
                                  padding=padding)

        self.deconv_A=BasicTConv2d(in_channels=int(32*n),out_channels=int(32*n),kernel_size=(1,4),stride=(1,2),padding=(0,1))
        # # self.deconv_A1 = BasicTConv2d(in_channels=int(32 * n), out_channels=int(32 * n), kernel_size=(1, 2),
        #                              stride=(1, 2), padding=(0, 0))
        self.deconv_B = BasicTConv2d(in_channels=int(32*n), out_channels=int(16*n), kernel_size=(1, 4), stride=(1, 2),padding=(0,1))

        self.conv_E=nn.Conv2d(in_channels=int(16*n),out_channels=out_channels,kernel_size=(1,5),stride=(1,1),padding=(0,2))

    def forward(self,x):
        xa=self.conv_A(x)
        # xa=self.dropout_a1(xa)
        xb=self.conv_B(xa)

        xc=self.deconv_A(xb)

        x_f=self.deconv_B(xc+xa)

        x_f=self.conv_E(x_f+x)
        x=x_f
        return x








class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.5)
        # self.batch=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x=self.batch(x)
        return x



class BasicTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding):
        super(BasicTConv2d, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.5)
        # self.batch=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.tconv(x)
        x = self.relu(x)
        # x=self.batch(x)
        return x