import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Trainer, TrainingArguments
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import AutoModelForCausalLMWithValueHead

from ..utils.general_utils import device_selector


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)


class GenerativeModel:
    def __init__(self, model_id: str, device: str = None, load_dtype: str = 'fp32', generation_config: GenerationConfig = None, accelerator_kwargs: dict = {},) -> None:
        """Wrapper class for all the generative models from 🤗 HuggingFace

        Args:
            model_id (str): model id or path from 🤗 Hugging Face
            device (str, optional): ['cpu', 'cuda', 'mps', None]. Defaults to None.
        """

        self.model_id = model_id
        self.accelerator = Accelerator(**accelerator_kwargs)

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

        self.load_dtype = load_dtype
        self.device = device

        if self.device is None:
            self.device = self.accelerator.device


        
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
        self.model = prepare_model_for_int8_training(self.model)    # double check
        self.model = get_peft_model(self.model, lora_config)
    

    def wrap_valueHead(self):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        self.model.gradient_checkpointing_disable = self.model.pretrained_model.gradient_checkpointing_disable
        self.model.gradient_checkpointing_enable = self.model.pretrained_model.gradient_checkpointing_enable


    def fine_tune(
            self,
            torch_dataset: torch.utils.data.Dataset, 
            optimized: bool = False,
            lr: float = 2e-4,
        ):
        """fine tune the model with the data provided

        Args:
            torch_dataset (torch.utils.data.Dataset): dataset containing the training data (torch Dataset, already tokenized)
            optimized (bool, optional): True for 8-bit and LoRA optimization (PEFT, Perforcance-efficient fine-tuning). Defaults to False.
        
            reference notebook from huggingface: https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=MDqJWba-tpnv
        """
        if optimized:
            assert self.load_dtype == '8-bit', '8 bit mode required for PEFT (optimized)'
        

        # apply some post-processing on the 8-bit model to enable training
        # freeze all layers and cast the layer-norm (and the output) to float32 for stability
        for param in self.model.parameters():
            param.requires_grad = False     # freeze
            if param.ndim == 1 and optimized:
                param.data = param.data.to(torch.float32)


        self.model.config.use_cache = False                # silence the warnings. Please re-enable for inference!
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()     # Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

        
        # cast final tensor logits to torch.float32
        setattr(
            self.model,
            list(self.model.named_children())[-1][0],    # name of the attribute
            CastOutputToFloat(getattr(self.model, list(self.model.named_children())[-1][0]))
        )

        if optimized:
            self.apply_LoRA()
        self.print_trainable_parameters()

        ## Training
        trainer = Trainer(
            model = self.model,
            train_dataset = torch_dataset,
            args = TrainingArguments(
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 4,
                warmup_steps = 100,
                max_steps = 200,
                optim = 'adamw_torch',
                learning_rate = lr,
                fp16 = True if optimized else False,
                logging_steps = 1,
                output_dir = './checkpoints/fine_tune/',
            ),
            data_collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm = False)
        )
        
        trainer.train()
        self.model.config.use_cache = True         # re-enable for inference
    
    def inference_fine_tuned(self, tokenized_batch: dict, return_decoded: bool = False):
        """Use the trained model to generate

        Args:
            tokenized_batch (dict): input_ids and attention_mask for model input
            return_decoded (bool, optional): True returns a string instead of the model output. Defaults to False.

        Returns:
            model_output: raw output of the model (to be decoded by the tokenizer) or str if `return_decoded == True`
        """


        if self.optimized:
            with torch.cuda.amp.autocast():
                output_model = self.model.generate(**tokenized_batch, generation_config = self.generation_config)
        else:
            with torch.no_grad():
                output_model = self.model.generate(**tokenized_batch, generation_config = self.generation_config)
    
        if return_decoded:
            self.tokenizer.decode(output_model[0], skip_special_tokens = True)
        
        return output_model

        
        
