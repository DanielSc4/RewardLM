# 🦖 **RL-on-LM**
Repo to keep code about the training of a RL algo on Transformer based LM 


## **ToxicityMeter**
Toxicity meter allows measuring the toxicity of generative models based on the output of a classifier ([RoBERTa for hate speech](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target))

Basic usage:
- Installation of required packages
    ```bash
    pip install -r requirements.txt
    ```

- Selection of models to be used by 🤗 HuggingFace to generate text and measure toxicity respectively
    ```Python
    model_id = 'MBZUAI/LaMini-GPT-1.5B'
    reward_model_id = 'facebook/roberta-hate-speech-dynabench-r4-target'
    ```

- Select the inference mode and enter the custom prompt:
    ```Python
    load_dtype = '8-bit'
    toxicity_meter = ToxicityMeter(
        model_id = model_id,
        load_dtype = 'fp32', # can be ['8-bit', 'bf16',]
        toxicity_model_id = reward_model_id,
    )

    custom_prompt = ('Donald Trump once said: "{prompt}')

    loader = gen_data(
        tokenizer = toxicity_meter.tokenizer, 
        max_len = 128 + len(custom_prompt),
        custom_prompt = custom_prompt,
        batch_size = 8,
    )

    toxicity_meter.measure_toxicity(loader = loader)
    ```