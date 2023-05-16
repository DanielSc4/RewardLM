import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import AutoModelForCausalLMWithValueHead

from ..utils.general_utils import device_selector

class GenerativeModel:
    def __init__(self, model_id: str, device: str = None, load_dtype: str = 'fp32', generation_config: GenerationConfig = None) -> None:
        """Wrapper class for all the generative models from 🤗 HuggingFace

        Args:
            model_id (str): model id or path from 🤗 Hugging Face
            device (str, optional): ['cpu', 'cuda', 'mps', None]. Defaults to None.
        """

        self.model_id = model_id
    

        if generation_config is None:
            self.generation_config = GenerationConfig(
                max_new_tokens = 25,
                num_beams = 5,
                early_stopping = True,
                pad_token_id = 0,       # crashes while using batchsize > 1 only on mps device if not set
                temperature = 0.8,
                top_p = .8,
                # diversity_penalty = .1, # should use num_beam_groups > 1
            )
        else:
            self.generation_config = generation_config

        if device is None:
            self.device = device_selector()
        else:
            assert load_dtype != '8-bit', '"8-bit" mode cannot be used with devices other than "cuda"'
            self.device = device
        
        if load_dtype == '8-bit':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map = 'auto',
                load_in_8bit = True,
            )
        elif load_dtype == 'bf16':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype = torch.bfloat16,
            ).to(self.device)
        else:
            # load in standard mode: float32
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)   

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def print_trainable_parameters(self) -> None:
        """Prints the number of trainable parameters in the model
        """
        train_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        
        print(f'trainable params: {train_params} || all params {all_params} || trainable(%): {train_params / all_params * 100:.2f}')


    def apply_LoRA(self, ):
        ## Required a peft model (ensure to have one using get_peft_model fun from peft)
        lora_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            target_modules = None, # handled automatically by peft
            lora_dropout = .05,
            bias = 'none',
            task_type = 'CAUSAL_LM',
        )
        self.model = prepare_model_for_int8_training(self.model, output_embedding_layer_name="embed_out")
        self.model = get_peft_model(self.model, lora_config)
    
    def wrap_valueHead(self):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        self.model.gradient_checkpointing_disable = self.model.pretrained_model.gradient_checkpointing_disable
        self.model.gradient_checkpointing_enable = self.model.pretrained_model.gradient_checkpointing_enable

