# https://happytransformer.com/text-generation/

########################
## SETUP              ##
########################

from happytransformer import HappyGeneration, GENEvalArgs
import torch
import os

# Turn of CUDA (for large fine-tuned modes that won't fit on GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args = GENEvalArgs(preprocessing_processes=1, mlm_probability=0.9)
# Data Location
evalData = r"Data\statisticslines.txt"

########################
## EVALUATIO          ##
########################

# Generic models (not fine-tuned)

# happy_gen = HappyGeneration("GPT2", "EleutherAI/gpt-j-6B")  
# resultGenericGPTJ6B = happy_gen.eval(evalData, args=args)

happy_gen = HappyGeneration()  
resultGeneric = happy_gen.eval(evalData, args=args)

happy_gen = HappyGeneration("GPT2", "gpt2-xl")
resultGenericGPT2XL = happy_gen.eval(evalData, args=args)

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M")  
resultGenericGPTNeo125M = happy_gen.eval(evalData, args=args)

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")  
resultGenericGPTNeo13B = happy_gen.eval(evalData, args=args)

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")  
resultGenericGPTNeo27B = happy_gen.eval(evalData, args=args)



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

print("**** EVAL RESULTS ****")
print("Generic GPT2:" + str(resultGeneric.loss))                        # EvalResult(loss=3.1945)
print("Generic GPT2-XL:" + str(resultGenericGPT2XL.loss))               # EvalResult(loss=2.7335)
print("Generic GPT-NEO-125M:" + str(resultGenericGPTNeo125M.loss))      # EvalResult(loss=2.9889)
print("Generic GPT-NEO-1.3B:" + str(resultGenericGPTNeo13B.loss))       # EvalResult(loss=2.5649)
print("Generic GPT-NEO-2.7B:" + str(resultGenericGPTNeo27B.loss))       # EvalResult(loss=2.4492)
print("Fine-tuned GPT-NEO-125M:" + str(resultGPTNeo125m.loss))          # EvalResult(loss=2.8519)
print("Fine-tuned GPT-NEO-1.3B:" + str(resultGPTNeo13B.loss))           # EvalResult(loss=2.0999)
print("Fine-tuned GPT-NEO-2.7B:" + str(resultGPTNeo27B.loss))           # EvalResult(loss=0.0001)