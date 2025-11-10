######################################## Rectangular Self-Calibration Module [ECCV-24] start ########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ============================================================
# 1. PyramidPoolAgg_PCE
# ============================================================
class PyramidPoolAgg_PCE(nn.Module):
    """
    Pyramid Pool Aggregation for Pyramid Context Extraction (PCE)

    Function:
        - Takes a list of inputs from different scales: inputs = [x1, x2, ..., xn]
        - Adaptive average pooling is applied to match the spatial size.
        - Features are concatenated along the channel dimension.
    
    Purpose:
        - Implements cross-scale feature fusion (similar to FPN fusion),
          facilitating unified context modeling for downstream tasks.
    """
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride  # Controls the downsampling factor

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape  # Use the spatial resolution of the last input as reference
        # Calculate target size (downscaled by stride)
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        # Apply adaptive average pooling to all input layers to match the size
        pooled = [F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs]
        # Concatenate along the channel dimension → [B, sum(C_i), H, W]
        return torch.cat(pooled, dim=1)


# ============================================================
# 2. ConvMlp
# ============================================================
class ConvMlp(nn.Module):
    """
    MLP implemented with 1x1 Convs that keep spatial dimensions unchanged.
    Source: Channel mixing module from the timm library.
    
    Function:
        - Similar to the FFN layer in Transformer.
        - Expands channels → activation → reduces channels
        - Does not change H, W dimensions
    """
    def __init__(
            self, 
            in_features,                # Input channels
            hidden_features=None,       # Hidden layer channels
            out_features=None,          # Output channels
            act_layer=nn.ReLU,          # Activation function type
            norm_layer=None,            # Optional normalization layer
            bias=True,
            drop=0.                     # Dropout ratio
        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Channel expansion (1x1 convolutions, preserving spatial size)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        # Optional normalization layer (default: Identity)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        # Activation function
        self.act = act_layer()
        # Dropout
        self.drop = nn.Dropout(drop)
        # Channel reduction (output convolution)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ============================================================
# 3. RCA: Rectangular Calibration Attention
# ============================================================
class RCA(nn.Module):
    """
    RCA Module: Rectangular Calibration Attention
    Function:
        - Combines local detail convolutions (square kernel)
        - With learnable directional attention (rectangular kernel)
        - Used for adaptive calibration of spatial dependencies (horizontal/vertical) in feature maps

    Core Innovation:
        - Uses learnable rectangular kernels (1×k, k×1)
        - More flexible and shape-adaptive compared to fixed kernels like StripPooling
    """
    def __init__(
        self, 
        inp,                    # Input channels
        kernel_size=1,          # Placeholder, unused
        ratio=2,                # Channel compression ratio (inp//ratio)
        band_kernel_size=11,    # Long side size of rectangular convolution kernel
        dw_size=(1,1),          # Placeholder, unused
        padding=(0,0),          # Placeholder
        stride=1,               # Stride
        square_kernel_size=3,   # Square convolution kernel size (used for local convolutions)
        relu=True               # Whether to use ReLU activation
    ):
        super(RCA, self).__init__()

        # --- (1) Local feature extraction ---
        # Depthwise 3x3 convolution, each channel conv separately (groups=inp)
        self.dwconv_hw = nn.Conv2d(
            inp, inp, square_kernel_size, 
            padding=square_kernel_size // 2,
            groups=inp
        )

        # --- (2) Directional global pooling ---
        # Compress along the width direction (retain vertical structure)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # Compress along the height direction (retain horizontal structure)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Channel compression dimension
        gc = inp // ratio

        # --- (3) Directional attention generation ---
        # Learnable rectangular convolutions (1×k, k×1)
        self.excite = nn.Sequential(
            # Horizontal convolution, receptive field (1, k)
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size),
                      padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            # Vertical convolution, receptive field (k, 1)
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1),
                      padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()  # Output attention weights (0~1)
        )

    def sge(self, x):
        """Spatial Gather & Excite"""
        # Get global features from horizontal and vertical directions
        x_h = self.pool_h(x)   # [B, C, H, 1]
        x_w = self.pool_w(x)   # [B, C, 1, W]
        # Combine the features from both directions
        x_gather = x_h + x_w
        # Generate attention weights through the excite network
        ge = self.excite(x_gather)  # [B, C, H, W]
        return ge

    def forward(self, x):
        # Local spatial features
        loc = self.dwconv_hw(x)
        # Global directional attention
        att = self.sge(x)
        # Fuse both by multiplication (weighted attention)
        out = att * loc
        return out


# ============================================================
# 4. RCM: Rectangular Calibration Meta Block
# ============================================================
class RCM(nn.Module):
    """
    RCM Module = RCA + Norm + ConvMLP + Residual
    A hybrid CNN-Transformer block.

    Args:
        dim (int): Input channels
        drop_path (float): DropPath probability
        ls_init_value (float): Layer Scale initialization value
    """
    def __init__(
            self,
            dim,
            token_mixer=RCA,             # Spatial mixing function (default RCA)
            norm_layer=nn.BatchNorm2d,   # Normalization layer type
            mlp_layer=ConvMlp,           # Channel mixing MLP type
            mlp_ratio=2,                 # MLP expansion ratio
            act_layer=nn.GELU,           # Activation function
            ls_init_value=1e-6,          # LayerScale parameter initialization value
            drop_path=0.,                # Stochastic depth drop rate
            dw_size=11,                  # band_kernel_size passed to RCA
            square_kernel_size=3,        # square_kernel_size passed to RCA
            ratio=1                      # Channel compression ratio passed to RCA
    ):
        super().__init__()

        # --- (1) Spatial mixing (RCA) ---
        self.token_mixer = token_mixer(
            dim, band_kernel_size=dw_size,
            square_kernel_size=square_kernel_size,
            ratio=ratio
        )

        # --- (2) Channel normalization ---
        self.norm = norm_layer(dim)

        # --- (3) Channel mixing MLP ---
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)

        # --- (4) Layer Scale parameter ---
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None

        # --- (5) DropPath regularization ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        # RCA: Spatial attention modeling
        x = self.token_mixer(x)
        # Normalization
        x = self.norm(x)
        # Channel mixing MLP
        x = self.mlp(x)
        # Layer Scale scaling
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        # DropPath + Residual connection
        x = self.drop_path(x) + shortcut
        return x


# ============================================================
# 5. multiRCM
# ============================================================
class multiRCM(nn.Module):
    """
    Stack of multiple RCA layers for enhanced feature calibration depth.
    Default stack of n=3 RCA blocks.
    """
    def __init__(self, dim, n=3) -> None:
        super().__init__()
        self.mrcm = nn.Sequential(
            *[RCA(dim, 3, 2, square_kernel_size=1) for _ in range(n)]
        )
    
    def forward(self, x):
        return self.mrcm(x)


# ============================================================
# 6. PyramidContextExtraction (PCE)
# ============================================================
class PyramidContextExtraction(nn.Module):
    """
    Multi-scale context extraction module.
    Combines PyramidPoolAgg + RCA.
    Function:
        - First, applies pyramid pooling to aggregate features at different scales.
        - Then uses RCA to capture global directional context.
        - Finally, splits the result according to the original channel proportions.
    """
    def __init__(self, dim, n=3) -> None:
        super().__init__()
        self.dim = dim  # List of input channels for each layer, e.g., [64, 128, 256]
        self.ppa = PyramidPoolAgg_PCE()  # Multi-scale pooling aggregation
        # n-layer RCA stacking for directional context modeling
        self.rcm = nn.Sequential(
            *[RCA(sum(dim), 3, 2, square_kernel_size=1) for _ in range(n)]
        )
        
    def forward(self, x):
        # Multi-scale feature fusion
        x = self.ppa(x)
        # Directional context enhancement
        x = self.rcm(x)
        # Split the output according to the original channels
        return torch.split(x, self.dim, dim=1)


# ============================================================
# 7. FuseBlockMulti
# ============================================================
class FuseBlockMulti(nn.Module):
    """
    A block to fuse low-level and high-level features (dual input version).

    Inputs:
        x = [x_low, x_high]
        x_low: [B, C, H, W] Low-level features
        x_high: [B, C, h, w] High-level features

    Flow:
        1. Apply 1x1 convolution to both inputs
        2. Upsample high-level output to match low-level size, after sigmoid activation
        3. Fuse the features by element-wise multiplication (attention gating)
    """
    def __init__(self, inp: int) -> None:
        super(FuseBlockMulti, self).__init__()

        # Apply feature transformation to both inputs
        self.fuse1 = Conv(inp, inp, act=False)
        self.fuse2 = Conv(inp, inp, act=False)
        self.act = h_sigmoid()  # Activation for attention gating

    def forward(self, x):
        x_l, x_h = x
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)      # Low-level feature convolution
        sig_act = self.fuse2(x_h)  # High-level feature convolution
        # Upsample high-level attention and activate
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        # Attention-weighted fusion
        out = inp * sig_act
        return out


# ============================================================
# 8. DynamicInterpolationFusion
# ============================================================
class DynamicInterpolationFusion(nn.Module):
    """
    Dynamic dual feature fusion module.
    - Matches the channel sizes by upsampling + 1x1 conv
    - Adds high-level features to low-level features for fusion
    """
    def __init__(self, chn) -> None:
        super().__init__()
        # chn = [low-level channels, high-level channels]
        self.conv = nn.Conv2d(chn[1], chn[0], kernel_size=1)
    
    def forward(self, x):
        # x = [x_low, x_high]
        return x[0] + self.conv(F.interpolate(
            x[1], size=x[0].size()[2:], mode='bilinear', align_corners=False
        ))

######################################## Rectangular Self-Calibration Module [ECCV-24] end ########################################
