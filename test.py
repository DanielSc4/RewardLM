import inseq

seq_model = inseq.load_model(
    'gpt2', # model,
    attribution_method='input_x_gradient'
)

out = seq_model.attribute(
    'hello world',
    step_scores=["probability"],
)

out.show()
