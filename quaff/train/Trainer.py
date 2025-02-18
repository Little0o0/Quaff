import os
import torch
from torch import nn
import functools
from tqdm import tqdm
from transformers import (
    Seq2SeqTrainer,
)
import warnings
import numpy as np
import json
import datetime
from datasets import load_dataset

class S2Strainer:
    def __init__(self, model, tokenizer, training_args, data_module, model_name=None, dataset_name=None):
        self.args = training_args
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = model.device
        self.trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        )
        # self.predict_dataset = data_module["predict_dataset"]
        if self.args.do_calibration:
            self.cal_dataloader = self.trainer.get_eval_dataloader()
    
    def train(self):
        return self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()

    def predict(self, dataset, output_dir):
        predictions = []
        prediction_ids = []
        label_ids = []
        eval_dataloader = self.trainer.get_eval_dataloader()
        for step, inputs in tqdm(enumerate(eval_dataloader)):
            loss, logits, labels = self.trainer.prediction_step(self.trainer.model, inputs, False)
            labels = labels.cpu()
            pred_ids = torch.argmax(logits, dim=-1).cpu()
            pred_ids = np.where(labels != -100, pred_ids, self.trainer.tokenizer.pad_token_id)
            preds = self.trainer.tokenizer.batch_decode(
                    pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

            prediction_ids += pred_ids.tolist()
            predictions += preds
            label_ids += labels.numpy().tolist()
            
        # current_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S-")
        # with open(os.path.join(output_dir, current_time+'predictions.jsonl'), 'w') as fout:
        #     for i, example in enumerate(dataset):
        #         data = {}
        #         data["input"] = example["input"]
        #         data["output"] = example["output"]
        #         data["prediction"] = predictions[i]
        #         data["prediction_ids"] = prediction_ids[i]
        #         data["label_ids"] = label_ids[i]
        #         fout.write(json.dumps(data) + '\n')
        return predictions
