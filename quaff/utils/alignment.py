
def peft_name_to_model_name(peft_layer_name, peft_type):

    name = peft_layer_name.replace("base_model.", "")
    if peft_type == "ia3":
        name = name.replace("model.model.", "model.").replace("model.lm_head", "lm_head")
    elif peft_type == "lora":
        if "lora" in peft_layer_name:
            return None
        name = name.replace("model.model.", "model.").replace("model.lm_head", "lm_head").replace(".base_layer", "")
    else:
        pass

    return name