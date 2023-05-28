# 🥞 **RewardLM**
Reward a Language Model with pancakes 🥞


## **Usage**
### 💀 ToxicityMeter

Toxicity meter allows measuring the toxicity of generative models based on the output of a classifier ([RoBERTa for hate speech](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target))


`...TBD...`

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








