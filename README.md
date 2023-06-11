# 🥞 **RewardLM**
Reward a Language Model with pancakes 🥞

## `TODO`:
- [ ] Catch & handle `ValueError: Responses are too short. Make sure they are at least 4 tokens long.` error skipping current batch that generates the anomaly.
- [ ] Add support for checkpointing and tracking [more info](https://huggingface.co/docs/accelerate/usage_guides/tracking).
- [ ] Add support for dynamic batch size based on [Memory Utilities](https://huggingface.co/docs/accelerate/usage_guides/memory) from 🤗 HuggingFace.
- [x] [fix] Fix short responses behavior (less than 4 tokens) [fix based on `generation_config`, TODO: how the generation change w/ bigger models?]
- [x] Add support for model's sharing (and backup) on 🤗 HuggingFace hub!
- [ ] Add possibility of using a reward manager as a reward model, to have more control over the reward system.
- [ ] Compatibility of ⚖️ ToxicityMeter with other datasets (possibly instructional).
- [ ] Extend ⚖️ ToxicityMeter compatibility with 🤗 Accelerate.
- [ ] Extend the possibility of managing parameters and configurations to 🥞RLAF.
- [ ] Use of [Inseq](https://github.com/inseq-team/inseq) for analysis and interpretability of generative models at ⚖️ ToxicityMeter.


## **Usage**
This repository gathers three main modules. Their operation is shared, allowing the training of any generative model following the two main techniques of Reinforcement Learning w/ PPO (🥞 RLAF) and the more classical 👨🏼‍🏫 fine-tune using PEFT techniques. The third module, ⚖️ Toxicity Meter, deals with measuring the toxicity of the responses of the generative model, whether pre-trained or after the 🥞 or 👨🏼‍🏫 process.


### **🥞 Reinforcement Learning with Automatic Feedback (RLAF)**


This module allows the use of reinforcement learning algorithms (specifically [PPO](https://openai.com/research/openai-baselines-ppo)) to optimise models according to a direction decided by the reward model. The process is similar to [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Reinforcement Learning from Human Feedback) but removes the human component from the loop to automate the process.

To 🥞 Reward a generative LM using the DIALCONAN dataset:

1. Select the generative and reward models you intend to use and other hyperparameters:
```python
import torch
from rewardlm.core.RL.RLModel import RLModel

rlmanager = RLModel(
    model_id = 'EleutherAI/pythia-70m',
    reward_model_id = 'facebook/roberta-hate-speech-dynabench-r4-target',
    optimized = True,   # use 8-bit PEFT
    # log_method = 'wandb',
    bs = 256,
    # force the use of CPU on Apple Silicon devices (mps not supported):
    accelerator_kwargs = {
        'cpu': False if torch.cuda.is_available() else True,
    },
)
```

2. Download the original dataset using the built in preprocessing functions:
```python
from rewardlm.data.data_utils import get_DIALOCONAN_prepro

data = get_DIALOCONAN_prepro(delete_last_assistant_response = True)
dataset = rlmanager.generate_dataset(text = data)
```

3. Start the PPO learning algorithm:
```python
history = rlmanager.train_PPO(dataset = dataset)
```


### **👨🏼‍🏫 Model fine-tune**
Each generative model can be fine-tuned on the same data used for Reinforcement Learning. In this way, it is possible to compare the results obtained from both techniques.

To fine-tune a generative model using the DIALCONAN dataset:

1. Select the model you intend to use and the `GenerativeModel` to get the use it:
```python
import torch
from rewardlm.core.GenerativeModel import GenerativeModel

model_id = 'facebook/opt-350m'
generator_manager = GenerativeModel(
    model_id,
    load_dtype = '8-bit' if torch.cuda.is_available() else 'fp32',
    # force the use of CPU on Apple Silicon devices (mps not supported):
    accelerator_kwargs = {
        'cpu': False if torch.cuda.is_available() else True,
    },
)
```

2. Download the original dataset using the built in preprocessing functions:

```python
from rewardlm.data.data_utils import get_DIALOCONAN_prepro
from rewardlm.data.CustomDatasets import PromptDataset_CLM

data = get_DIALOCONAN_prepro()

dataset = PromptDataset_CLM(
    tokenizer = generator_manager.tokenizer,
    text = data,
    custom_prompt = custom_prompt,
)
```

3. Start the fine-tutning process:
```python
generator_manager.fine_tune(
    torch_dataset = dataset, 
    optimized = True if torch.cuda.is_available() else False,
)
```


### **⚖️ ToxicityMeter**

Toxicity meter allows measuring the toxicity of generative LM based on the output of a classifier ([RoBERTa for hate speech](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target))


1. Choose `model_id`, `batchsize` and (opt) generation parameters:
```python
from transformers import GenerationConfig
from rewardlm.ToxicityMeter import ToxicityMeter
from rewardlm.utils.general_utils import device_selector

model_id = 'bigscience/bloomz-3b'
reward_model_id = 'facebook/roberta-hate-speech-dynabench-r4-target'
batchsize = 16
generation_config = GenerationConfig(
    max_new_tokens = 25,
    num_beams = 5,
    early_stopping = True,
    # crashes while using batchsize > 1 on mps device if not set:
    pad_token_id = 0,
    temperature = 0.8,
    top_p = .8,
    # diversity_penalty = .1, # should use num_beam_groups > 1
)

load_dtype = 'bf16' # can be ['8-bit', 'bf16', 'fp32']
toxicity_meter = ToxicityMeter(
    model_id = model_id,
    load_dtype = load_dtype,
    toxicity_model_id = reward_model_id,
    device = device_selector(),
    generation_config = generation_config,
)
```

2. Customize your prompt from the original dataset and generate the `toxicity_df` dataset:
```python
from rewardlm.data.data_utils import get_real_toxicity_prompts

custom_prompt = ('User: "{prompt}".\nAssistant: ')
# custom prompts required by the original paper of RedPajama
if model_id == 'togethercomputer/RedPajama-INCITE-Chat-3B-v1':
    custom_prompt = ('<human>: "{prompt}"\n<bot>: ')

toxicity_df = toxicity_meter.measure_toxicity(
    text_prompt = get_real_toxicity_prompts()['text'].to_list(),
    custom_prompt = custom_prompt, 
    generation_config = generation_config,
    batch_size = batchsize,
)
```

3. Save the obtained results:
```python
# save csv in tmp folder
fld = './result analysis/tmp'
toxicity_df.to_csv(
    fld + f'/measured_tox_instruct_{model_id.split("/")[-1]}_{load_dtype}.csv'
)
```




## Tested Models and datasets
#### **Generative language models:**
- `LaMini-LM`: Small-sized collection of efficient language models distilled from ChatGPT and trained on a large-scale dataset of 2.58M instructions [GitHub](https://github.com/mbzuai-nlp/LaMini-LM/), [Paper](https://arxiv.org/abs/2304.14402)

- `RedPajama-*`: [Source](https://www.together.xyz/blog/redpajama-models-v1)

- `BloomZ`: Family of models capable of following human instructions in dozens of languages zero-shot [GitHub](https://github.com/bigscience-workshop/xmtf), [Paper](https://arxiv.org/abs/2211.01786)

- `Pythia`: *Predominantly abandoned in favour of instructed models.* Model(s) that combines interpretability analysis and scaling laws to understand how knowledge develops and evolves during training in autoregressive transformers. [GitHub](https://github.com/EleutherAI/pythia), [Paper](https://arxiv.org/pdf/2304.01373.pdf)

#### **Datasets:**
- `Real Toxicity Prompts`: Mainly used for the ⚖️ `ToxicityMeter` module. Dataset of 100K naturally occurring, sentence-level prompts derived from a large corpus of English web text, paired with toxicity scores from a widely-used toxicity classifier. [GitHub](https://github.com/allenai/real-toxicity-prompts), [Paper](https://www.semanticscholar.org/paper/RealToxicityPrompts%3A-Evaluating-Neural-Toxic-in-Gehman-Gururangan/399e7d8129c60818ee208f236c8dda17e876d21f)

- `DIALOCONAN`: Mainly used for 👨🏼‍🏫 `fine-tuning` and 🥞 `RLAF` modules. Datasets of counter-narratives to fight online hate speech. [GitHub](https://github.com/marcoguerini/CONAN#dialoconan), [Paper](https://arxiv.org/pdf/2211.03433v1.pdf)

#### **Reward models:**
- `roberta-hate-speech-dynabench-r4-target`: Model trained on ∼40,000 entries, generated and labelled by trained annotators over four rounds
of dynamic data creation. [Paper](https://arxiv.org/pdf/2012.15761.pdf)





## **Development**
### How to setup on [Google Colab](https://colab.research.google.com/):
1. Import the main notebook in colab
2. Include the following cell at the beginning:
```bash
!git clone https://__TOKEN_GIT__:@github.com/DanielSc4/RewardLM.git
%cd RewardLM/
!pip install -r requirements.txt
from huggingface_hub import login
login(token = '__TOKEN_HF__')
```
3. [Opt, only if the repo is private] Replace `__TOKEN_GIT__` with your git token ([more info here](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token))
4. Replace `__TOKEN_HF__` with you 🤗 HuggingFace personal [token](https://huggingface.co/settings/tokens)

### How to setup developer environment

**Dependency install:**
1. Install `poetry`, a Python package manager
2. It is recommended to run the following command to let `poetry` create the virtual environment for the project
directly inside the root folder, allowing IDEs to detect dependencies and executables
```bash
poetry config virtualenvs.in-project true
```
3. Inside the root folder, run `poetry install` to get all the dependencies. See 
[Poetry docs](https://python-poetry.org/docs/) for a thorough explanation of how poetry works

**Activating virtual env:**

To run a project file, you will need to use the interpreter installed by Poetry in the virtual environment, usually located
in `rewardlm/.venv/bin/`. To do that, you can use `poetry run` command, followed by the name of the script that you 
want to run ([Poetry run doc](https://python-poetry.org/docs/cli#run)).

You can also run the following command to ensure that the terminal will use the correct python version (the one downloaded in the
virtual env) together with its whole set of dependencies:
```bash
source .venv/bin/activate
```








