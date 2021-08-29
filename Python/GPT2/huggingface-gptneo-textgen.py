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
sentenceStart = 'Statistics problems can help make decisions.'

#baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller GPT-Neo model
#baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger GPT-Neo model
baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Largest GPT-Neo model
baseModelArchitecture = "gpt"
fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
else :
    fineTunedModelLocation = fineTunedModelLocation + baseModelArchitecture


# Parameters for text generation
maxLength = 200
topK = 500
temperature = 0.7
topProbabilities = 0.92
numberOfSentenceSequences = 4

# Generate text on pre-trained model
generator = pipeline(task="text-generation", model=baseModelArchitecture, device=deviceId, framework="pt", use_fast=False)
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
generator = pipeline(task="text-generation", model=fineTunedModelLocation, device=deviceId, framework="pt", use_fast=False)
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