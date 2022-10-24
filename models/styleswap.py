import math
import itertools

import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.op.fused_act import FusedLeakyReLU
from models.op.fused_act import fused_leaky_relu
from models.op.upfirdn2d import upfirdn2d

isconcat = True

class PixelNorm(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
       
        return input * paddle.rsqrt(paddle.mean(input * input, axis=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = paddle.to_tensor(k, dtype='float32')
    if k.ndim == 1:
        k = k.unsqueeze(0) * k.unsqueeze(1)
    k /= k.sum()
    return k


class Upfirdn2dUpsample(nn.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor * factor)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=self.factor,
                        down=1,
                        pad=self.pad)

        return out


class Upfirdn2dDownsample(nn.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input,
                        self.kernel,
                        up=1,
                        down=self.factor,
                        pad=self.pad)

        return out


class Upfirdn2dBlur(nn.Layer):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor * upsample_factor)

        self.register_buffer("kernel", kernel)#, persistable=False)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualConv2D(nn.Layer):
    """This convolutional layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        self.weight = self.create_parameter(
            (out_channel, in_channel, kernel_size, kernel_size),
            default_initializer=nn.initializer.Normal())
        self.scale = 1 / math.sqrt(in_channel * (kernel_size * kernel_size))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = self.create_parameter((out_channel, ),
                                              nn.initializer.Constant(0.0))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            "{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            " {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLinear(nn.Layer):
    """This linear layer class stabilizes the learning rate changes of its parameters.
    Equalizing learning rate keeps the weights in the network at a similar scale during training.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()
        
        self.weight = self.create_parameter(
            (in_dim, out_dim), default_initializer=nn.initializer.Normal())
        self.weight.set_value((self.weight / lr_mul))
        if bias:
            self.bias = self.create_parameter(
                (out_dim, ), nn.initializer.Constant(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
       
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input,
                           self.weight * self.scale,
                           bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            "{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]})"
        )

class ScaledLeakyReLU(nn.Layer):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope
        # self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)

class ModulatedConv2D(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Upfirdn2dBlur(blur_kernel,
                                      pad=(pad0, pad1),
                                      upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * (kernel_size * kernel_size)
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = self.create_parameter(
            (1, out_channel, in_channel, kernel_size, kernel_size),
            default_initializer=nn.initializer.Normal())

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            "{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            "upsample={self.upsample}, downsample={self.downsample})")

    def forward(self, in_im, style):
        batch, in_channel, height, width = in_im.shape

        # style = self.modulation(style).reshape((batch, 1, in_channel, 1, 1))
        style = self.modulation(style)
        style = style.reshape((batch, 1, in_channel, 1, 1))
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = paddle.rsqrt((weight * weight).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape((batch, self.out_channel, 1, 1, 1))

        weight = weight.reshape((batch * self.out_channel, in_channel,
                                 self.kernel_size, self.kernel_size))

        if self.upsample:
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            weight = weight.reshape((batch, self.out_channel, in_channel,
                                     self.kernel_size, self.kernel_size))
            weight = weight.transpose((0, 2, 1, 3, 4))
            weight = weight.reshape(
                (batch * in_channel, self.out_channel, self.kernel_size,
                 self.kernel_size))
            out = F.conv2d_transpose(in_im,
                                     weight,
                                     padding=0,
                                     stride=2,
                                     groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))
            out = self.blur(out)

        elif self.downsample:
            in_im = self.blur(in_im)
            _, _, height, width = in_im.shape
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(in_im, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        else:
            in_im = in_im.reshape((1, batch * in_channel, height, width))
            out = F.conv2d(in_im, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape((batch, self.out_channel, height, width))

        return out


class NoiseInjection(nn.Layer):
    def __init__(self):
        super().__init__()

        self.weight = self.create_parameter(
            (1, ), default_initializer=nn.initializer.Constant(0.0))

    def forward(self, image, noise=None):
        if noise is not None:
            ## print(image.shape, noise.shape)
            if isconcat: return paddle.concat([image, self.weight * noise], 1) # concat
            return image + self.weight * noise

        if noise is None:
            batch, _, height, width = image.shape
            noise = paddle.randn((batch, 1, height, width))
        
        if isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image * (1 - self.weight) + self.weight * noise
        # return image + self.weight * noise

class ConstantInput(nn.Layer):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = self.create_parameter(
            (1, channel, size, size),
            default_initializer=nn.initializer.Normal())

    def forward(self, input):
       
        batch = input.shape[0]
        out = self.input.tile((batch, 1, 1, 1))

        return out

class StyledConv(nn.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2D(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel * feat_multiplier)


    def forward(self, input, style, noise=None):
       
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # # print("input in style conv", out.shape)
        out = self.activate(out)

        return out


class ToRGB(nn.Layer):
    def __init__(self,
                 in_channel,
                 style_dim,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upfirdn2dUpsample(blur_kernel)

        self.conv = ModulatedConv2D(in_channel,
                                    3,
                                    1,
                                    style_dim,
                                    demodulate=False)
        self.bias = self.create_parameter((1, 3, 1, 1),
                                          nn.initializer.Constant(0.0))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class ToOutterMask(nn.Layer):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upfirdn2dUpsample(blur_kernel)

        self.conv = ModulatedConv2D(in_channel, 1, 1, style_dim, demodulate=False)

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        # out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip
            
        out = F.sigmoid(out)

        return out

class Generator(nn.Layer):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        outter_mask=True,
        narrow=1

    ):
        super().__init__()
        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)
        self.outter_mask = outter_mask

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4],
                                self.channels[4],
                                3,
                                style_dim,
                                blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4]*self.feat_multiplier, style_dim, upsample=False)
        
        if outter_mask:
            self.to_outtermask1 = ToOutterMask(self.channels[4]*self.feat_multiplier, style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        # self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.LayerList()
        self.upsamples = nn.LayerList()
        self.to_rgbs = nn.LayerList()
        self.to_outtermasks = nn.LayerList()
        # self.noises = nn.Layer()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel*self.feat_multiplier,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                ))

            self.convs.append(
                StyledConv(out_channel*self.feat_multiplier,
                           out_channel,
                           3,
                           style_dim,
                           blur_kernel=blur_kernel))

            self.to_rgbs.append(ToRGB(out_channel*self.feat_multiplier, style_dim))
            if outter_mask:
                self.to_outtermasks.append(ToOutterMask(out_channel*self.feat_multiplier, style_dim))
            else:
                self.to_outtermasks.append(None)
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [paddle.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(paddle.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = paddle.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def conver_noise_withmask(self, noise, mask):
        att_msk = mask
        masked_img = noise * att_msk
        color_sum = paddle.sum(masked_img, (2, 3))
        mask_sum = paddle.sum(att_msk, (2, 3))
        ratio = color_sum / mask_sum
        ratio = ratio.unsqueeze(2).unsqueeze(3)
        ratio_mask = ratio * att_msk + (1 - att_msk) * noise
        return ratio_mask


    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        noise_blank_level=9
        ):
        outtermasks = []
        outtermask = None
        if not input_is_latent:
            styles = [self.style(s) for s in styles]
        if noise is None:
            '''
            noise = [None] * (2 * (self.log_size - 2) + 1)
            '''
            noise = []
            batch = styles[0].shape[0]
            for i in range(self.n_mlp + 1):
                size = 2 ** (i+2)
                noise.append(paddle.randn((batch, self.channels[size], size, size)))

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).tile((1, inject_index, 1))
            latent2 = styles[1].unsqueeze(1).tile((1, self.n_latent - inject_index, 1))

            latent = paddle.concat([latent, latent2], 1)

       
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        
        skip = self.to_rgb1(out, latent[:, 1])
        if self.outter_mask:
            outtermask = self.to_outtermask1(out, latent[:, 1])
            outtermasks.append(outtermask)
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb, to_outtermask in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs, self.to_outtermasks
            ):
            if i > noise_blank_level and self.outter_mask:
                blankmask = F.interpolate(outtermasks[-1], noise1.shape[-2:])
                noise1 = self.conver_noise_withmask(noise1, blankmask)
                noise2 = self.conver_noise_withmask(noise2, blankmask)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)

            skip = to_rgb(out, latent[:, i + 2], skip)
            if self.outter_mask:
                outtermask = to_outtermask(out, latent[:, i + 2], outtermask)
                outtermasks.append(outtermask)
            i += 2
  

        image = skip

        return image, outtermask

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Upfirdn2dBlur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2D(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            ))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Layer):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(in_channel,
                              out_channel,
                              1,
                              downsample=True,
                              activate=False,
                              bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class FullGenerator(nn.Layer):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        outter_mask=True,
        narrow=1,
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
            }
        self.outter_mask = outter_mask
        self.log_size = int(math.log(size, 2))
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, \
            isconcat=isconcat, 
            outter_mask=outter_mask,
            narrow=narrow)
        
        conv = [ConvLayer(3, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(style_dim, style_dim, activation='fused_lrelu'))
    
    def forward(self,
        inputs,
        id_emb,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        ):
        noise = []
        target_img = inputs.clone()
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        id_emb = self.final_linear(id_emb)
        outs = [id_emb]

        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]

        out_img, mask = self.generator(outs, return_latents, inject_index, truncation, truncation_latent, 
            input_is_latent, noise=noise[1:])
        if self.outter_mask:
            return out_img * mask + (1 - mask) * target_img
        else:
            return out_img