# Performance-Efficient Fine-Tuning using Reinforcement Learning
# Adaptation of https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import prepare_model_for_int8_training

class PerfRLModel:
    def __init__(self, model_id, reward_model_id) -> None:
        
        # loading model in 8-bit
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit = True, device_map = 'auto')
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model_tokenizer.padding_side = "left" 
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token

        if "gpt-neox" in model_id:
            model = prepare_model_for_int8_training(
                model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"]
            )
        else:
            model = prepare_model_for_int8_training(model)
