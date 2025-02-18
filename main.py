import sys
import os

import evaluate

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from quaff.train.Trainer import S2Strainer
from quaff.nn.model import build_model, build_peft_model, build_quantized_model
from quaff.train.Datasets import make_data_module
from quaff.utils.arguments import get_args
from quaff.calibration.outlier_detection import detect_outlier_channel
import torch
from os.path import exists, join, isdir
import logging
import warnings
import numpy as np 
import random 
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO,)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "quaff" # name your W&B project
    args, training_args = get_args()
    setup_seed(args.seed)
    model, tokenizer = build_model(args)
    logging.info("############ model structure before setting peft ############")
    logging.info(model)
    model_name = args.model_name_or_path.split("/")[-1]

    outlier_dict = None  # for quaff only
    if args.do_calibration:
        logging.info(f"######### start calibrate with public dataset {args.cal_dataset_name} (identify idx of the outlier channels) ########")
        data_module = make_data_module(tokenizer=tokenizer, args=args, calibration=True)
        
        trainer = S2Strainer(model, tokenizer, training_args, data_module, model_name, args.dataset_name)
        outlier_dict = detect_outlier_channel(trainer.trainer, trainer.cal_dataloader, detection_strategy=args.outlier_strategy, threshold=args.outlier_threshold, maximum_ratio=args.outlier_max_ratio)

        os.makedirs(args.outlier_output_dir, exist_ok=True)
        logging.info(f"######### save  the idx of the outlier channels ########")
        torch.save(outlier_dict, os.path.join(args.outlier_output_dir, f"{args.outlier_strategy}{args.outlier_threshold}_{args.cal_dataset_name}_{model_name}.pt"))
    
    else:
        outlier_dict = torch.load( os.path.join(args.outlier_output_dir, f"{args.outlier_strategy}{args.outlier_threshold}_{args.cal_dataset_name}_{model_name}.pt") )
    

    model = build_quantized_model(model, outlier_dict=outlier_dict,)
    logging.info("############ model structure after quant ############")
    logging.info(model)
    
    ### build peft model
    model, tokenizer = build_peft_model(args, model, tokenizer)
    logging.info("############ model structure after setting peft ############")
    logging.info(model)

    ##### test the saving function
    if args.do_train or args.do_eval or args.do_predict:
        data_module = make_data_module(tokenizer=tokenizer, args=args, calibration=False)
        trainer = S2Strainer(model, tokenizer, training_args, data_module, model_name, args.dataset_name)
        print(args)
    
    if args.do_train:
        train_result = trainer.train()
    
    if args.do_eval:
        train_result = trainer.evaluate()

    if args.do_predict:
        trainer.predict(data_module["predict_dataset"], args.output_dir)


