# 🥞 **RewardLM**
Reward a Language Model with pancakes 🥞


## **Usage**
### ⚖️ ToxicityMeter

Toxicity meter allows measuring the toxicity of generative LM based on the output of a classifier ([RoBERTa for hate speech](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target))


`...TBD...`

### 🥞 Reinforcement Learning with Automatic Feedback (RLAF)

`...TBD...`

### 👨🏼‍🏫 Model fine-tune
Each generative model can be fine-tuned on the same data used for Reinforcement Learning. In this way, it is possible to compare the results obtained from both techniques.

To fine-tune a generative model using the DIALCONAN dataset:

1. Select the model you intend to use and the `GenerativeModel` to get the use it:
```python
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
from rewardlm.data.data_utils import get_DIALOCONAN_for_finetune
from rewardlm.data.CustomDatasets import PromptDataset_CLM

data = get_DIALOCONAN_for_finetune()

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
!git clone https://__TOKEN_GIT__:@github.com/DanielSc4/RL-on-LM.git
%cd RL-on-LM/
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








