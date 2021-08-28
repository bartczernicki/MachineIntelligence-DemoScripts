# https://happytransformer.com/text-generation/

########################
## SETUP              ##
########################

from transformers import pipeline
import torch

# Set seed for reproducability, only for non-deterministic steps
seed = 400
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Determine if to use GPU or CPU for compute
deviceId = 0 if torch.cuda.is_available() else -1

################################
## TEXT GENARATION            ##
################################
sentenceStart = 'Statistics problems can be solved using resampling.'
baseModel = "EleutherAI/gpt-neo-1.3B" # Generic pre-trained model
baseModelFineTuned = r"Models\HappyTransformer-FineTuning-TextGen-27B" # Fine-tuned model

# Parameters for text generation
maxLength = 100
topK = 500
temperature = 0.75
topProbabilities = 0.9
numberOfSentenceSequences = 4

# Generate text on pre-trained model
generator = pipeline(task="text-generation", model=baseModel, device=deviceId, framework="pt", use_fast=False)
generatorResults = generator(
    sentenceStart,
    do_sample=True, 
    max_length=maxLength,
    top_k=topK,
    temperature=temperature,
    top_p=topProbabilities, 
    num_return_sequences=numberOfSentenceSequences
)
print(generatorResults)

# Generate text on fine-tuned model
generator = pipeline(task="text-generation", model=baseModelFineTuned, device=deviceId, framework="pt", use_fast=False)
generatorResults = generator(
    sentenceStart,
    do_sample=True,
    max_length=maxLength,
    top_k=topK,
    temperature=temperature,
    top_p=topProbabilities, 
    num_return_sequences=numberOfSentenceSequences
)
print(generatorResults)