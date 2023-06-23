import torch
import torch.nn as nn
import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, Trainer, TrainingArguments
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import AutoModelForCausalLMWithValueHead


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float16) # ex float32


class GenerativeModel:
    def __init__(self, model_id: str, load_from_peft: bool = False, device: str = None, load_dtype: str = 'fp32', generation_config: GenerationConfig = None, accelerator_kwargs: dict = {},) -> None:
        """Wrapper class for all the generative models from 🤗 HuggingFace

        Args:
            model_id (str): model id or path from 🤗 Hugging Face or id containing LoraConfig if load_from_peft = True
            device (str, optional): ['cpu', 'cuda', 'mps', None]. Defaults to None.
        """

        self.model_id = model_id
        self.accelerator = Accelerator(**accelerator_kwargs)
        print(f'Accelerator selected device: {self.accelerator.device}')

        if generation_config is None:
            self.generation_config = GenerationConfig(
                max_new_tokens = 256,
                min_new_tokens = 4,
                num_beams = 4,
                early_stopping = True,
                # pad_token_id = 0,       # crashes while using batchsize > 1 only on mps device if not set
                temperature = 0.8,
                top_p = .75,
                top_k = 40
                # diversity_penalty = .1, # should use num_beam_groups > 1
            )
        else:
            self.generation_config = generation_config

        self.load_dtype = load_dtype
        self.device = device

        if self.device is None:
            self.device = self.accelerator.device


        if load_from_peft:
            config = LoraConfig.from_pretrained(self.model_id)
            self.original_pretrained_model_id = config.base_model_name_or_path
            print(f'Obtaining original model: {self.original_pretrained_model_id}')
            self.__load_from_peft(config, load_dtype)
        else:
            if load_dtype == '8-bit':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto" if torch.cuda.is_available() else 'cpu',
                    # device_map = self.device,
                    torch_dtype = torch.float16,
                    load_in_8bit = True,
                )
                print(f'Model loaded in {load_dtype} mode')
            elif load_dtype == 'bf16':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype = torch.bfloat16,
                ).to(self.device)
                print(f'Model loaded in {load_dtype} mode')
            else:
                # load in standard mode: float32
                self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
                print(f'Model loaded in fp32 (standard) mode')

        # tokenizer
        if load_from_peft:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.original_pretrained_model_id, 
                # use_fast = False,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"        # Allow batched inference
        
        # TODO: check where is best to set pad_token id to 0 or pad_token = eos_token!
        # Setting to 0 from alpaca-lora finetune script
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def __load_from_peft(self, config, load_dtype: str):
        # function to load a pretrained finetuned w/ peft model from huggingface, usign the original model and the specified configuration
        original_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(self.original_pretrained_model_id)
        self.model = get_peft_model(model = original_pretrained, peft_config=config)


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


    def apply_LoRA(self, prepare: bool):
        ## Required a peft model (ensure to have one using get_peft_model fun from peft)
        lora_config = LoraConfig(
            r = 32,
            lora_alpha = 32,
            target_modules = None, # handled automatically by peft
            lora_dropout = .05,
            bias = 'none',
            task_type = 'CAUSAL_LM',
        )
        if prepare:
            self.model = prepare_model_for_int8_training(self.model)    # if prepare is False, the preprocessing is done before
        self.model = get_peft_model(self.model, lora_config)
    

    def wrap_valueHead(self):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model)

        self.model.gradient_checkpointing_disable = self.model.pretrained_model.gradient_checkpointing_disable
        self.model.gradient_checkpointing_enable = self.model.pretrained_model.gradient_checkpointing_enable


    def fine_tune(
            self,
            dataset: datasets.Dataset, 
            val_set_per: float = .0,
            optimized: bool = True,
            lr: float = 3e-4,
            epochs: int = 3,
            initial_bs: int = 32,
            run_name: str = 'random_name',
        ):
        """fine tune the model with the data provided

        Args:
            dataset (torch.utils.data.Dataset): dataset containing the training data (torch Dataset, already tokenized)
            val_set_per (float, optional): percentage (%) of the dataset used for evaluation. Defaults to 0 (%)
            optimized (bool, optional): True for 8-bit and LoRA optimization (PEFT, Perforcance-efficient fine-tuning). Defaults to False.
        
            reference notebook from huggingface: https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing#scrollTo=MDqJWba-tpnv
        """

        # manual preparing the model to 8-bit training
        prepare = False
        if not prepare:
            # apply some post-processing on the 8-bit model to enable training
            # freeze all layers and cast the layer-norm (and the output) to float32 (or 16) for stability
            # TODO: check the following code!
            for param in self.model.parameters():
                param.requires_grad = False     # freeze all parameters
                if param.ndim == 1 and optimized:
                    param.data = param.data.to(torch.float32)       # ex float 32
                    pass


            self.model.config.use_cache = False             # silence the warnings. Re-enable for inference!
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()         # Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.

            # cast final tensor logits to torch.float32 (or float16)
            setattr(
                self.model,
                list(self.model.named_children())[-1][0],    # name of the attribute
                CastOutputToFloat(getattr(self.model, list(self.model.named_children())[-1][0]))
            )

        if optimized:
            self.apply_LoRA(prepare = prepare)
        self.print_trainable_parameters()

        # spliting dataset if eval
        if val_set_per > 0:
            assert val_set_per < 1, f'val_set_per should be less than 1 ([0 - 99]%), {val_set_per} given'

            train_eval_d = dataset.train_test_split(
                test_size = int(len(dataset) * val_set_per), shuffle=True, seed=42,
            )
            train_dataset, val_dataset = train_eval_d['train'], train_eval_d['test']

        else:
            train_dataset = dataset
            val_dataset = None

        ## Training
        trainer = Trainer(
            model = self.model,
            train_dataset = train_dataset,
            eval_dataset = val_dataset, 
            args = TrainingArguments(
                per_device_train_batch_size = initial_bs,       # initial batchsize set
                gradient_accumulation_steps = 2,                # (gradient_acc_steps * initial_bs = total_batchsize)   # not working w/ > 1
                warmup_steps = 100,
                num_train_epochs = epochs,
                learning_rate = lr,
                fp16 = True if torch.cuda.is_available() else False,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_per > 0 else "no",
                eval_steps=200 if val_set_per > 0 else None,    # perform evaluation every _ step 
                save_strategy="steps",
                save_steps=200,
                auto_find_batch_size = True,            # lower batchsize exp to avoid CUDA out of memory
                use_mps_device = False,                 # torch.backends.mps.is_available(),    # NOT working even if False!
                logging_strategy="steps",
                logging_steps=10,
                save_total_limit = 4,       # max number of checkpoints saved (delete the older one)
                output_dir = './checkpoints/fine_tune/',
                report_to='wandb',
                run_name = run_name,
            ),
            data_collator = transformers.DataCollatorForSeq2Seq(
                self.tokenizer, 
                # mlm = False, 
                pad_to_multiple_of = 8,
                return_tensors='pt', 
                padding = True,
            )
        )
        print(trainer.accelerator)
        print(type(trainer.accelerator))
        print(f'Trainer device: {trainer.accelerator.device}')
        print(f'Trainer args device: {trainer.args.device}')
        
        # with torch.autocast("cuda"):
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
        # if self.optimized:
        #     with torch.cuda.amp.autocast():
        #         output_model = self.model.generate(**tokenized_batch, generation_config = self.generation_config)
        # else:

        # solves error w/ missing pad_token
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        for ele in tokenized_batch:
            tokenized_batch[ele] = tokenized_batch[ele].to(self.accelerator.device)
        
        if not hasattr(self.model.config, 'quantization_config'):
            # not loaded in 4 or 8 bit mode, else leave model as is
            self.model.to(self.accelerator.device)
            # if not self.model.config.to_dict()['quantization_config']['load_in_8bit']:
            
        self.model.eval()
        with torch.no_grad():
            output_model = self.model.generate(**tokenized_batch, generation_config = self.generation_config)

        if return_decoded:
            output_model = self.tokenizer.decode(output_model[0], skip_special_tokens = True)

        return output_model


    def push_to_hub(self, repo_id: str):
        self.model.push_to_hub(repo_id)
        print('https://huggingface.co/' + repo_id)
    