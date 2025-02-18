import bitsandbytes as bnb
from torch import nn
from quaff.autograd._function import MatMul8bitQuaff, MatmulState_Quaff
import bitsandbytes.functional as F
import torch

class Linear8bitQuaff(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, outlier_idx=None, S_momentum=False, device=None, weight_quant_type="linear", act_quant_type="vector", grad_precision=8, layer_name=""):
        super().__init__(input_features, output_features, bias, device)

        # discard the full precision weight to save the memory
        self.weight = nn.parameter.Parameter(None, requires_grad=False)
        self.weight.to(device)

        self.state = MatmulState_Quaff()
        self.state.outlier_idx = [] if outlier_idx is None else outlier_idx
        self.state.S = torch.ones(input_features).to(device)

        self.state.S_momentum = S_momentum
        self.state.weight_quant_type = weight_quant_type
        self.state.act_quant_type = act_quant_type
        self.state.layer_name = layer_name
        self.state.grad_precision = grad_precision

        self.device = device

        self.W_int = None
        self.Delta_W = None
        self.Wo = None
    
    def to(self, device, *args, **kwargs):
        self.device = device
        if self.W_int is not None:
            self.W_int = self.W_int.to(device, *args, **kwargs)
        if self.Delta_W is not None:
            self.Delta_W = self.Delta_W.to(device, *args, **kwargs)
        if self.Wo is not None:
            self.Wo = self.Wo.to(device, *args, **kwargs)
        
    @staticmethod
    def load_from_linear(module, outlier_idx, S_momentum=False, weight_quant_type="linear", act_quant_type="vector", grad_precision=8, layer_name=""):
        assert isinstance(module, torch.nn.Linear)

        if outlier_idx is None:
            outlier_idx = []

        new_module = Linear8bitQuaff(
            module.in_features, module.out_features, module.bias is not None, outlier_idx,
            S_momentum, module.weight.device, weight_quant_type, act_quant_type, grad_precision, layer_name)

        if len(module.weight.data.shape) == 2:
            dim = 0
        else:
            dim = 1

        

        Wo_data = module.weight.data[..., outlier_idx].T.contiguous()
        W_int_data, Delta_W_data = F.vectorwise_quant(module.weight.data.T, dim=dim, quant_type=weight_quant_type)
        
        new_module.W_int = nn.parameter.Parameter(W_int_data, requires_grad=False).contiguous()
        new_module.Delta_W = nn.parameter.Parameter(Delta_W_data, requires_grad=False).contiguous()
        new_module.Wo = nn.parameter.Parameter(Wo_data, requires_grad=False).contiguous()

        if module.bias is not None:
            new_module.bias = module.bias

        return new_module

    def forward(self, X: torch.Tensor):
        if self.bias is not None and self.bias.dtype != X.dtype:
            self.bias.data = self.bias.data.to(X.dtype)

        if self.Wo.dtype != X.dtype:
            self.Wo.data = self.Wo.data.to(X.dtype)

        if self.Delta_W.dtype != X.dtype:
            self.Delta_W.data = self.Delta_W.data.to(X.dtype)

        if self.state.S is not None:
            if self.state.S.dtype != X.dtype:
                self.state.S.data = self.state.S.data.to(X.dtype)
            if self.state.S.device != X.device:
                self.state.S = self.state.S.to(X.device)

        if self.state.So is not None:
            if self.state.So.dtype != X.dtype:
                self.state.So.data = self.state.So.data.to(X.dtype)
            if self.state.So.device != X.device:
                self.state.So = self.state.So.to(X.device)
        
        out = MatMul8bitQuaff.apply(X, self.W_int, self.Delta_W, self.Wo, self.state)

        if self.bias is not None:
            out = out + self.bias

        return out