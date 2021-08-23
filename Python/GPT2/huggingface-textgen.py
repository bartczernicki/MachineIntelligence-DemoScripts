# 1) Install PyTorch
# https://pytorch.org/get-started/locally/
# 2) Install Huggingface Transformers
# pip3 install transformers

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
import torch
import numpy as npcle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# encode context the generation is conditioned on
input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))