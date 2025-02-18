from typing import Any
import warnings
import torch
import bitsandbytes.functional as F
from bitsandbytes.autograd._functions import MatmulLtState, undo_layout
class MatmulState_Quaff(MatmulLtState):

    S = None # tensor
    So = None  # tensor, only include the scaling factor for outlier idx
    S_momentum = False # boolean, if use the momentum method to adjust the S
    outlier_idx = [] # the index of the outlier channel
    layer_name = "" # denote the layer name of the model
    weight_quant_type = "linear" # vector or linear , see bitsandbytes.functional.vectorwise_quant
    act_quant_type = "vector" # vector or linear , see bitsandbytes.functional.vectorwise_quant
    grad_precision = 8 # gradient bit width
    using_igemmlt = False
    Wo = None # weights in outlier channels
    SCBt = None
    p = 0.5 # S = max(X)^p max(W)^(1-p).
    beta = 0.9 # this is for the S momentum
    quant_outlier = False # if quant outlier, we find there is not obvious acceleration for outlier quantizaiton 

    def __init__(self):
        super().__init__()

    def update_S(self, S_input):
        if self.S_momentum and self.S is not None:
            self.S = self.beta * S_input + (1- self.beta) * self.S
        else:
            self.S = S_input
        return self.S

    def update_S_outlier(self, S_outlier_input):
        if self.S_momentum and self.So is not None:
            self.So = self.beta * S_outlier_input + (1- self.beta) * self.So
        else:
            self.So = S_outlier_input

        self.So = self.So.to(self.S.device)
        self.S[self.outlier_idx] = self.So
        return self.S, self.So


class MatMul8bitQuaff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_int, Delta_W, Wo, state=MatmulState_Quaff()):
        # W: output \times input

        # 1. Get(or Update) scaling factor s based on predefined outlier_idx
        # 2. Quantize Scaled \hat{X} = Xs^{-1} to produce \hat{X}_{int} and \Delta_{\hat{X}}
        # 3. Matmul \hat{X}_{int} with W_{int} and dequant the result with \Delta_{\hat{X}} and \Delta_W
        # 4. Quantize Scaled \hat{w} =  (s-1)Wo to produce \hat{w}_{int} and \Delta_\hat{w}
        # 5. Matmul   \hat{X}_{int}[:, outlier_idx] and \hat{w}_{int} and dequant the result with \Delta_{\hat{X}} and \Delta_\hat{w}
        # 6. Add two results

        outlier_idx = state.outlier_idx
        act_quant_type = state.act_quant_type
        quant_outlier = state.quant_outlier

        assert len(outlier_idx) == Wo.shape[0]
        assert X.dtype == Wo.dtype

        # only support  W:[in_channel, out_channel]
        if len(X.shape) == 3:
            X_dims = [0, 1]
        else:
            X_dims = [0]

        assert len(W_int.shape) == 2

        # 1. Get(or Update) scaling factor S based on predefined outlier_idx
        # if S is None or So is None:
        max_X = torch.amax(torch.abs(X[..., outlier_idx]), dim=X_dims)
        max_W = torch.amax(torch.abs(Wo), dim=1)
        So = torch.maximum(torch.sqrt(max_X / max_W), torch.ones_like(max_W))
        S, So = state.update_S_outlier(So)

        # 2. Quantize Scaled X: XS^{-1} to produce hatX_int and Delta_X
        # We want to use X[..., outlier_idx] /= So, but it will suffer from inplace error in grad, Therefore we use X /= S
        hatx = X[..., outlier_idx] / So
        # X[..., outlier_idx] = hatx
        # X.is_contiguous()
        hatX_int, Delta_X = F.vectorwise_quant(X / S, dim=-1, quant_type=act_quant_type)

        # 3. Matmul hatX_int with W_int and dequant the result with Delta_X and Delta_W
        normal_outputq = F.igemm(hatX_int, W_int)
        normal_output = F.vectorwise_mm_dequant(normal_outputq, Delta_X, Delta_W, X.dtype, act_quant_type)

        
        if len(outlier_idx) == 0:
            outlier_output = 0
        elif len(outlier_idx) % 4 == 0 and quant_outlier:
            if len(Wo.shape) == 2:
                dim = 0
            else:
                dim = 1
            scaled_Wo = (So.unsqueeze(-1) - 1) * Wo
            Wqo, Wo_s = F.vectorwise_quant(scaled_Wo, dim=dim, quant_type=state.weight_quant_type)
            outlier_outputq = F.igemm(hatX_int[..., outlier_idx], Wqo)
            outlier_output = F.vectorwise_mm_dequant(outlier_outputq, Delta_X, Wo_s, X.dtype, act_quant_type)
        else:
            outlier_output = torch.matmul(hatx, (So.unsqueeze(-1) - 1) * Wo)
        ########################## Temp #########################

        # 6. Add two results
        output = normal_output + outlier_output

        if X.requires_grad:
            ctx.save_for_backward(W_int, Delta_W, Wo)
        else:
            ctx.save_for_backward(None, None, None )
        ctx.state = state
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        W_int, Delta_W, Wo = ctx.saved_tensors
        state = ctx.state
        weight_quant_type = state.weight_quant_type
        act_quant_type = state.act_quant_type
        layer_name = state.layer_name
        grad_precision = state.grad_precision
        outlier_idx = state.outlier_idx
        grad_X = None

        grad_output = grad_output.contiguous()

        if W_int is not None:
            if len(grad_output.shape) == 3:
                dims = [2]
            else:
                dims = [1]

            if len(W_int.shape) == 3:
                # bio -> boi
                permute_dim = [0, 2, 1]
                dim_W = dims
            else:
                # io -> oi
                permute_dim = [1, 0]
                dim_W = [1]

            # the gradient quantization or not
            if grad_precision != 8:
                with torch.no_grad():
                    W = W_int * Delta_W / 127  # manual dequant
                    grad_X = torch.matmul(grad_output, W.permute(permute_dim))
            else:
                # k_proj should contiguous and I do not know the reason?
                if "k_proj" in layer_name and not qgrad_output.is_contiguous():
                    grad_output = grad_output.contiguous()
                qgrad_output, S1 = F.vectorwise_quant(
                    grad_output, dim=dims, quant_type=act_quant_type
                )

                if weight_quant_type != "linear":
                    with torch.no_grad():
                        W = W_int * Delta_W / 127  # manual dequant and requant
                        # W[outlier_idx,] = Wo

                    W_rq, W_rs = F.vectorwise_quant(W, dim=dim_W, quant_type=weight_quant_type)
                    igrad_X = F.igemm(qgrad_output, W_rq.permute(permute_dim))
                    grad_X = F.vectorwise_mm_dequant(
                        igrad_X,
                        S1,
                        W_rs.permute(permute_dim),
                        grad_output.dtype,
                        act_quant_type,
                    )
                else:
                    igrad_X = F.igemm(qgrad_output, W_int.permute(permute_dim))
                    grad_X = F.vectorwise_mm_dequant(
                        igrad_X,
                        S1,
                        Delta_W,
                        grad_output.dtype,
                        act_quant_type,
                    )

        return grad_X, None, None, None, None