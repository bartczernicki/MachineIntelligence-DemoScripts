# https://happytransformer.com/text-generation/

########################
## SETUP              ##
########################

from transformers import pipeline
import torch

# Set seed for reproducability, only for non-deterministic steps
seed = 300
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Determine if to use GPU or CPU for compute
deviceId = 0 if torch.cuda.is_available() else -1

################################
## TEXT GENARATION            ##
################################
sentenceStart = 'Statistics is important in baseball analytics.'
baseModel = "EleutherAI/gpt-neo-1.3B" # Generic pre-trained model
baseModelFineTuned = r"Models\HappyTransformer-FineTuning-TextGen" # Fine-tuned model

# Parameters for text generation
maxLength = 100
topK = 400
temperature = 0.7
topProbabilities = 0.92
numberOfSentenceSequences = 2

# Generate text on pre-trained model
generator = pipeline("text-generation", model=baseModel, device=deviceId)
generatorResults = generator(
    sentenceStart,
    do_sample=True, 
    max_length=maxLength,
    top_k=topK,
    temperature=temperature,
    top_p=topProbabilities, 
    num_return_sequences=2 
)
print(generatorResults)

# Generate text on fine-tuned model
generator = pipeline("text-generation", model=baseModelFineTuned, device=deviceId)
generatorResults = generator(
    sentenceStart,
    do_sample=True,
    max_length=maxLength,
    top_k=topK,
    temperature=temperature,
    top_p=topProbabilities, 
    num_return_sequences=2 
)
print(generatorResults)