
## SETUP
# 1) Install PyTorch, different for each OS/inference compute
# https://pytorch.org/get-started/locally/
# 2) Install Huggingface Transformers
# pip3 install transformers

## CODE
# import transformers
# from transformers import pipeline
# # Print version
# print("Transformers version: " + transformers.__version__)

# classifier = pipeline('sentiment-analysis')
# results = classifier(["We are very happy to show you the 🤗 Transformers library.",
#            "This code is absolultely positively terrible!"])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# # https://github.com/huggingface/notebooks/blob/master/course/chapter1/section3.ipynb
# generator = pipeline("text-generation", model="distilgpt2")
# genResults = generator(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=2,
# )
# for genResult in genResults:
#     print(genResult)

# https://huggingface.co/transformers/notebooks.html
# https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb

# COMMON SCRIPT SETUP
import torch
import numpy as npcle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set seed for reproducability, only for non-deterministic steps
torch.manual_seed(100)

# Load gpt2 model, can be gpt-large etc.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# Encode context the generation is conditioned on
input_ids = tokenizer.encode('I am a Chief Technical Officer.', return_tensors='pt')


## TEXT GENERATION - GPT2 - GREEDY
# Generate text until the output length (which includes the context length) reaches 75
greedy_output = model.generate(input_ids, max_length=75)

# Notice the text-gen output repeats itself
print("Greedy Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


## TEXT GENERATION - GPT2 - BEAM
# Generate text until the output length (which includes the context length) reaches 75
beam_output = model.generate(input_ids, max_length=75,
    # Beam, no repeat & early stopping
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
    )

# Notice the text-gen output does NOT repeats itself
print("Beam Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))


## TEXT GENERATION - GPT2 - BEAM
# Generate text until the output length (which includes the context length) reaches 75
beam_output = model.generate(input_ids, max_length=75,
    # Beam, no repeat & early stopping
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
    )

# Notice the text-gen output does NOT repeats itself
print("Beam Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))


## TEXT GENERATION - GPT2 - MORE BEAMS & MULTIPLE SEQUENCES
# Generate text until the output length (which includes the context length) reaches 75
beam_outputs = model.generate(input_ids, max_length=75,
    # Beam, no repeat & early stopping
    num_beams=20,
    no_repeat_ngram_size=2,
    early_stopping=True,
    # Multiple sentences returned
    num_return_sequences=5
    )

# Notice the text-gen output does NOT repeats itself (sequences are SLIGHTLY different), returns 5 sentences
print("Multiple Beams Outputs:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
  print("{}: {}".format(i+1, tokenizer.decode(beam_output, skip_special_tokens=True)))


## TEXT GENERATION - GPT2 - SAMPLING
# Generate text until the output length (which includes the context length) reaches 75
# use temperature to decrease the sensitivity to low probability candidates
sampling_outputs = model.generate(input_ids, 
    do_sample=True, 
    max_length=75, 
    top_k=50,
    temperature=0.7,
    # Multiple sentences returned
    num_return_sequences=5
)

# Notice the text-gen output does NOT repeats itself (sequences are VERY different), returns 5 sentences
print("Multiple Sampling Outputs:\n" + 100 * '-')
for i, sampling_output in enumerate(sampling_outputs):
  print("{}: {}".format(i+1, tokenizer.decode(sampling_output, skip_special_tokens=True)))


## TEXT GENERATION - GPT2 - SAMPLING & PROBABILTY
# Generate text until the output length (which includes the context length) reaches 75
# use temperature to decrease the sensitivity to low probability candidates
samplingProbability_outputs = model.generate(input_ids, 
    do_sample=True, 
    max_length=75, 
    top_k=50,
    temperature=0.7,
    top_p=0.9, 
    # Multiple sentences returned
    num_return_sequences=5
)

# Notice the text-gen output does NOT repeats itself (sequences are VERY DIFFERENT & more related), returns 5 sentences
print("Multiple Sampling & Probability Outputs:\n" + 100 * '-')
for i, samplingProbability_output in enumerate(samplingProbability_outputs):
  print("{}: {}".format(i+1, tokenizer.decode(samplingProbability_output, skip_special_tokens=True)))


## TEXT GENERATION - GPT2 - SAMPLING & PROBABILTY
# Generate text until the output length (which includes the context length) reaches 75
# use temperature to decrease the sensitivity to low probability candidates
samplingProbability_outputs = model.generate(input_ids, 
    do_sample=True, 
    max_length=75,
    # Mix of outputs
    top_k=400,
    temperature=0.6,
    top_p=0.9, 
    # Multiple sentences returned
    num_return_sequences=10
)

# Notice the text-gen output does NOT repeats itself (mix of parameters for realistic output), returns 5 sentences
print("Multiple Sampling & Probability Outputs:\n" + 100 * '-')
for i, samplingProbability_output in enumerate(samplingProbability_outputs):
  print("{}: {}".format(i+1, tokenizer.decode(samplingProbability_output, skip_special_tokens=True)))


## TEXT GENERATION - GPT2-LARGE - SAMPLING & PROBABILTY
# Info on model sizes: https://huggingface.co/transformers/pretrained_models.html
# Load gpt2 model, can be gpt-large etc.
largeTokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
# Add the EOS token as PAD token to avoid warnings
largeModel = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
# Encode context the generation is conditioned on
input_ids = largeTokenizer.encode('You should learn statistical analysis.', return_tensors='pt')

# Generate text until the output length (which includes the context length) reaches 75
# use temperature to decrease the sensitivity to low probability candidates
samplingProbability_outputs = largeModel.generate(input_ids, 
    do_sample=True, 
    max_length=75,
    # Mix of outputs
    top_k=400,
    temperature=0.6,
    top_p=0.9, 
    # Multiple sentences returned
    num_return_sequences=10
)

# Notice the text-gen output does NOT repeats itself (mix of parameters for realistic output), returns 5 sentences
print("Multiple Sampling & Probability Outputs:\n" + 100 * '-')
for i, samplingProbability_output in enumerate(samplingProbability_outputs):
  print("{}: {}".format(i+1, tokenizer.decode(samplingProbability_output, skip_special_tokens=True)))