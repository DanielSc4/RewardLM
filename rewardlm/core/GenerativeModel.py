import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.general_utils import device_selector

class GenerativeModel:
    def __init__(self, model_id: str, device: str = None, load_dtype: str = 'fp32') -> None:
        """Wrapper class for all the generative models from 🤗 HuggingFace

        Args:
            model_id (str): model id or path from 🤗 Hugging Face
            device (str, optional): ['cpu', 'cuda', 'mps', None]. Defaults to None.
        """

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
    
    