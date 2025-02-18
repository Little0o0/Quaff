import torch
from collections import Counter
from torch import nn
import functools
from tqdm import tqdm
from collections import defaultdict

def find_prefix(prefixs, layer_name):
    for p in prefixs:
        if p in layer_name:
            return p
    return "others"


def detect_outlier_channel(trainer, cal_dataloader, detection_strategy="mean_times",  maximum_ratio=0.01, threshold:float=100.):
    # this function is used to detect the outlier channel for the Quaff
    # strategy: support top_abs, static, zscore, mean_times
    # maximum_ratio: the max ratio of the number of outlier channels to the total number of input channels, it can be a dict :{"prefix_layer": ratio}, or a int for the unifrom distribution.
    # threshold : for the static detection, it is the magnitude threshold; for the zscore detection , it is the zscore threshold; for the meam_times detection, it is the scaling threshold for mean value 
    
    # trainer: transformer trainer
    # cal_dataloader : dataloader used to detect outlier

    outlier_idx = defaultdict(Counter) # {layer_name: outlier_Counter}
    num_channels = {} # {layer_name: num_input_channels}
    
    
    def stat_tensor(name, t, W, strategy):
        in_features = t.shape[-1]
        num_channels[name] = in_features
        t = t.view(-1, in_features)

        if isinstance(maximum_ratio, dict):
            ratio = maximum_ratio[find_prefix(maximum_ratio.keys(), name)]
        else:
            ratio = maximum_ratio
        
        if strategy == "static" and threshold is not None:
            bound  = torch.tensor(threshold).float()
            idx = (t.abs() > bound).nonzero().T[1].tolist()
            
        elif strategy == "top_abs":
            num_outlier_channels = int(in_features * ratio)
            value, _ = torch.max(t.abs(), dim=0)
            _, idx = value.topk(num_outlier_channels)
        elif strategy == "mean_times":
            bound = torch.mean(t.abs()) * threshold
            idx = (t.abs() > bound).nonzero().T[1].tolist()
        elif strategy == "zscore":
            mean, std = torch.mean(t), torch.std(t)
            z_score = (t - mean) / std 
            bound = torch.tensor(threshold).float()
            idx = (z_score.abs() > bound).nonzero().T[1].tolist()
        else:
            raise Exception(f"No detection methods named {strategy} ")
        
        idx_counter = Counter(idx)
        outlier_idx[name] = outlier_idx[name] + idx_counter

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        strategy = detection_strategy
        stat_tensor(name, x, m.weight.data.T, strategy)
    
    hooks = []
    for name, m in trainer.model.named_modules():
        if isinstance(m, nn.Linear):

            # we do not detect peft module
            if "lora" in name or "prompt" in name or "ia3" in name:
                continue

            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    # CUDA OOM !!!
    # _, _, _ = self.trainer.predict(self.trainer.train_dataset)

    for step, inputs in tqdm(enumerate(cal_dataloader)):
        _, _, _ = trainer.prediction_step(trainer.model, inputs, True)

    for h in hooks:
        h.remove()

    outlier_dict = {} # {name : index_list}
    for name in outlier_idx:
        if isinstance(maximum_ratio, dict):
            ratio = maximum_ratio[find_prefix(maximum_ratio.keys(), name)]
        else:
            ratio = maximum_ratio
        outlier_counter = outlier_idx[name]
        input_channels = num_channels[name]
        max_outliers = int(input_channels * ratio // 4 * 4) # must be {4,8,16 ...} for igemm
        outlier_list = [x[0] for x in outlier_counter.most_common(max_outliers)]
        outlier_dict[name] = outlier_list
    return outlier_dict         