# implementing the shape formula here: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
import torch
n_samples, n_channels, height, width= 2, 3, 77, 65
x= torch.randn(n_samples, n_channels, height, width)

m= torch.nn.Conv2d(in_channels= n_channels, out_channels= 17, kernel_size= (5,3), stride= (2,2))
y= m(x)

index= 2 # 2:heght 3: width

calc_height= torch.floor(torch.tensor(1+ (1/m.stride[index-2]) * (x.shape[index] - 2* m.padding[index-2] - m.dilation[index-2] *(m.kernel_size[index-2]-1)-1) ))

print(f'actual height: {y.shape[index]} calc height: {calc_height}')

#%%

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w