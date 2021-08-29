# https://happytransformer.com/text-generation/

########################
## SETUP              ##
########################

from happytransformer import HappyGeneration, GENEvalArgs
import torch
import os

# Turn of CUDA (for large modes that won't fit on GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args = GENEvalArgs(preprocessing_processes=1, mlm_probability=0.9)
# Data Location
evalData = r"Data\statisticslines.txt"

########################
## EVALUATIO          ##
########################

# Generic model (not fine-tuned)
happy_gen = HappyGeneration()  
resultGeneric = happy_gen.eval(evalData, args=args)


# Fine-Tuned model (GPT-NEO 125mil)
baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller GPT-Neo model
fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
else :
    fineTunedModelLocation = fineTunedModelLocation + baseModelArchitecture

happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=fineTunedModelLocation) 
resultGPTNeo125m = happy_gen.eval(evalData, args=args)

# Fine-Tuned model (GPT-NEO 1.3B)
baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Smaller GPT-Neo model
fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
else :
    fineTunedModelLocation = fineTunedModelLocation + baseModelArchitecture

happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=fineTunedModelLocation) 
resultGPTNeo13B = happy_gen.eval(evalData, args=args)

# Fine-Tuned model (GPT-NEO 2.7B)
baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Smaller GPT-Neo model
fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
else :
    fineTunedModelLocation = fineTunedModelLocation + baseModelArchitecture

happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=fineTunedModelLocation) 
resultGPTNeo27B = happy_gen.eval(evalData, args=args)

########################
## RESULTS            ##
########################

print("Generic GPT2:" + resultGeneric.loss)  # EvalResult(loss=X)
print("Fine-tuned GPT-NEO-125M:" + resultGPTNeo125m.loss)  # EvalResult(loss=X)
print("Fine-tuned GPT-NEO-1.3B:" + resultGPTNeo13B.loss)  # EvalResult(loss=X)
print("Fine-tuned GPT-NEO-2.7B:" + resultGPTNeo27B.loss)  # EvalResult(loss=X)