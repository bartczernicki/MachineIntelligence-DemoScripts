# Hugging Face - Transformers Course
# https://huggingface.co/course/chapter1/1?fw=pt

########################
## SETUP              ##
########################

from transformers import pipeline


################################
## ZERO-SHOT CLASSIFICATION   ##
################################
# This pipeline is called zero-shot because you don’t need to fine-tune the model on your data to use it. It can directly return probability scores for any list of labels you want!
classifier = pipeline("zero-shot-classification")
classifierResults = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(classifierResults)

################################
## TEXT GENARATION            ##
################################
generator = pipeline("text-generation", model="distilgpt2")
generatorResults = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(generatorResults)

################################
## MASK FILLING               ##
################################
unmasker = pipeline("fill-mask")
unmaskerResults = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(unmaskerResults)