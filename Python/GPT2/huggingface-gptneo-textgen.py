# https://happytransformer.com/text-generation/

########################
## SETUP              ##
########################

from transformers import pipeline


################################
## TEXT GENARATION            ##
################################
sentenceStart = 'You should consider learning statistics before approaching data science.'

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
generatorResults = generator(
    sentenceStart,
    do_sample=True, 
    max_length=75,
    top_k=400,
    temperature=0.6,
    top_p=0.9, 
    num_return_sequences=2 
)
print(generatorResults)