from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3unet arch
def upsample(x,fac=2):
    return F.interpolate(x, scale_factor=fac, mode='bilinear')
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x
class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
        ChannelAttention(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
class PhaseNet(nn.Module):
    def __init__(self, in_chn=4, wf=32, depth=4, relu_slope=0.2):
        super(PhaseNet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf),HinResBlock(wf,wf))
        self.pre_re = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf))
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(FFTConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_FFT_AMP=False, use_FFT_PHASE=True))
            prev_channels = (2**i) * wf
        # 8*wf-> wf
        # wf->2*wf
        self.up_path_1 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(HinResBlock((2**i)*wf,(2**i)*wf))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.demosaic = nn.Sequential(nn.Conv2d(wf,4*wf,3,1,1),nn.PixelShuffle(2))
        # self.last = conv3x3(prev_channels, in_chn, bias=True)
    # in_chn>> input channel
    def forward(self, x,demo):
        image = x #pack image
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                
                encs.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        # x1 32channel
        x1 = self.demosaic(x1)

        sam_feature, out_1 = self.sam12(x1, demo)


        #stage 2 : phase stage
        out_1_fft = torch.fft.rfft2(out_1, dim=(-2, -1))
        out_1_phase = torch.angle(out_1_fft)

        image_fft = torch.fft.rfft2(demo, dim=(-2, -1))
        image_phase = torch.angle(image_fft)
        image_amp =torch.abs(image_fft)
        image_inverse = torch.fft.irfft2(image_amp*torch.exp(1j*out_1_phase), dim=(-2, -1))

        return [out_1_phase,sam_feature,image_inverse,out_1,encs,decs]
    def get_input_chn(self, in_chn):
        return in_chn
class cas(nn.Module):
    # parmid color embedding

    def __init__(self,nc):
        super(cas, self).__init__()
        self.adj1 = CAB(nc)
        self.adj2 = CAB(2*nc)
        self.adj3 = CAB(4*nc)
        
    def forward(self, c, shortcuts):
        #shortcuts[0] 4
        #shortcuts[1] 2
        #shortcuts[2] 1
        x_1_color,c_1 = self.adj1(shortcuts[0],c)
        x_2_color,c_2 = self.adj2(shortcuts[1],c_1)
        x_3_color,_= self.adj3(shortcuts[2],c_2)
        
        return [x_1_color, x_2_color, x_3_color]


@ARCH_REGISTRY.register()
class FourierISP(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=4, relu_slope=0.2):
        super(FourierISP, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf),HinResBlock(wf,wf))
        self.conv_02 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf),HinResBlock(wf,wf))
        self.pre_re = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf))
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(FFTConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_FFT_AMP=True, use_FFT_PHASE=False))

            self.down_path_2.append(FFTConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_FFT_AMP=False, use_FFT_PHASE=False))
            prev_channels = (2**i) * wf
        # 8*wf-> wf
        # wf->2*wf
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(HinResBlock((2**i)*wf,(2**i)*wf))
            self.skip_conv_2.append(HinResBlock((2**i)*wf,(2**i)*wf))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = Refine(prev_channels,in_chn)
        self.cat123= nn.Conv2d(prev_channels*3, prev_channels, 1, 1, 0)
        self.last2 = Refine(wf,in_chn)
        self.p_net = PhaseNet(in_chn = in_chn+1,wf=wf)
        self.cas = cas(wf)

    def forward(self, x,demo):
        image = demo #Input image


        out_1_phase,sam_feature,image_inverse,p_net_out,pencs,pdecs = self.p_net(x,demo)

        #stage 1 : amplitude JPEG restore stage
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)  #before_down:x1, after_doen:x1_up
                encs.append(x1_up)
            else:
                x1 = down(x1)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)

        amplitude_sam_feature, amplitude_out_1 = self.sam12(x1, image)


        #stage 2 : amplitude stage

        amplitude_out_1_fft = torch.fft.rfft2(amplitude_out_1, dim=(-2, -1))
        amplitude_out_1_amp = torch.abs(amplitude_out_1_fft)




        x2 = self.conv_02(image_inverse)
        x2 = self.cat123(torch.cat([x2, sam_feature,amplitude_sam_feature], dim=1))
        blocks = []
        
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, upsample(pencs[i]), upsample(pdecs[-i-1])) #Color Adaptation
                blocks.append(x2_up) 
            else:
                x2 = down(x2)

        blocks = self.cas(amplitude_sam_feature,blocks)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1])) #应该用colorfeature来处理这个

        out_2 = self.last(x2)
        tmp_2 = out_2 + image
        out_2 = tmp_2
        out_2 = self.pre_re(tmp_2)
        out_2 = self.last2(out_2)+tmp_2
        #pre
        return [out_1_phase,amplitude_out_1_amp,out_2]
    def get_input_chn(self, in_chn):
        return in_chn

class FFTConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_FFT_PHASE=False, use_FFT_AMP=False):
        super(FFTConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff
        self.use_FFT_PHASE = use_FFT_PHASE
        self.use_FFT_AMP = use_FFT_AMP
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_fft_1 = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.relu_fft_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_fft_2 = nn.Conv2d(out_size, out_size, 1, 1, 0)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        if self.use_FFT_PHASE and self.use_FFT_AMP == False:
            x_res = self.identity(x)
            x_fft =torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_phase = self.conv_fft_1(x_phase)
            x_phase = self.relu_fft_1(x_phase)
            x_phase = self.conv_fft_2(x_phase)
            x_fft_res = torch.fft.ifft2(x_amp*torch.exp(1j*x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real
            out  = out + x_res + x_fft_res
        elif self.use_FFT_AMP and self.use_FFT_PHASE == False:
            x_res = self.identity(x)
            x_fft =torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_amp = self.conv_fft_1(x_amp)
            x_amp = self.relu_fft_1(x_amp)
            x_amp = self.conv_fft_2(x_amp)
            x_fft_res = torch.fft.ifft2(x_amp*torch.exp(1j*x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real
            out  = out + x_res + x_fft_res
        else:
            out = out + self.identity(x)

        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = FFTConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
class SFT(nn.Module):
    def __init__(self, nc):
        super(SFT,self).__init__()
        self.convmul = nn.Conv2d(nc,nc,3,1,1)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2d(2*nc, nc, 1, 1, 0)

    def forward(self, x, res):
        # res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(torch.cat([x,mul*x+add],1))
        return fuse

class FreBlockAdjust(nn.Module):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.sft = SFT(nc)
        self.cat = nn.Conv2d(2*nc,nc,1,1,0)

    def forward(self,x, y_amp, y_phase):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(torch.cat([y_phase,pha],1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out

class CAB(nn.Module):
    def __init__(self, in_nc):
        super(CAB,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = HinResBlock(in_nc,in_nc)
        self.frequency_process = FreBlockAdjust(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)
        self.transamp = nn.Conv2d(in_nc,in_nc,1,1,0)
        self.transpha = nn.Conv2d(in_nc,in_nc, 1, 1, 0)
        self.down = conv_down(in_nc, 2*in_nc, bias=False)

    def forward(self, x, color_feature):
        xori = x
        _, _, H, W = x.shape
        x_f = x
        x_freq = torch.fft.rfft2(x_f, norm='backward')
        color_feature_fre =  torch.fft.rfft2(color_feature, norm='backward')
        xout_fre_amp,xout_fre_phase = torch.abs(color_feature_fre), torch.angle(color_feature_fre)
        y_amp,y_phase = self.transamp(xout_fre_amp),self.transpha(xout_fre_phase)
        x_f = self.spatial_process(x_f)
        x_freq = self.frequency_process(x_freq, y_amp, y_phase)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x_f,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori,self.down(color_feature)

class ModulateUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(ModulateUNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = FFTConvBlock(in_size, out_size, False, relu_slope)
        self.process = CAB(out_size)

    def forward(self, x, bridge):
        up = self.up(x) 
        out = self.process(up,bridge)
        return out



class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True,inc=3):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, inc, kernel_size, bias=bias)
        self.conv3 = conv(inc, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img



if __name__ == '__main__':
    # pha = PhaseNet().cuda()
    src = torch.randn((1, 3, 448, 448)).cuda()
    x = torch.randn((1,4,224,224)).cuda()
    net = FourierISP(wf=24).cuda()
    out_1_phase,amplitude_out_1_amp,out_2 = net(x,src)
    print(out_1_phase.size())
