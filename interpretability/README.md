# Interpretability analysis

##### Implementation of analysis using the [🐛Inseq](https://github.com/inseq-team/inseq) intrepretability library.

The script `gen_attribute.py` generate the attribuition for a subsample of sentences. The goal is to compare the [low, high] toxicity responses produced by the fine-tuned and RLAF generative models, comparing them to the [low, high] responses by the original pretrained model.

Usage:

```bash
python interpretability/gen_attributes.py \
    -m configs/debug_GPT-neo.yaml \
    -i interpretability/interp_configs/i_debug_GPT-neo.yaml \
    -s 5 
```

- `-m` or `--model_config`: Config file (.yaml) of the model to test.
- `-i` or `--interp_config`: Config file (.yaml) for the interpretability script.
- `-s` or `--start_from` (*optional*):  Start from n-th iteration (used to jump ahead in the dataset in case of attribution already done in a previous backup). Defaults to 0.

Each attribution, as soon as it is calculated by [🐛Inseq](https://github.com/inseq-team/inseq), is aggregated by subwords and then compressed to a `G x T` matrix, where `G` represents the number of tokens generated and `T` the number of total tokens (prompt + generation). This operation is performed runtime to streamline the computational resources required in terms of allocated memory and fastness in storage (exponential demand otherwise).

___

It is possible at a later stage to explore the attribution obtained through the use of, for example, the prompt *dependancy metric* (see `get_prompt_dependancy` in `interp_utils.py`), which shows how much a generation relies on the prompt (or itself) during token generation.

*To be continued...*