# https://happytransformer.com/text-generation/

import huggingfacehelpers # Custom module helpers
from happytransformer import HappyGeneration, GENEvalArgs
import torch
import os

def main():

    ########################
    ## CONFIG             ##
    ########################

    # Config Variables
    seed = huggingfacehelpers.random.randint(1, 100000) # Set to static value instead of RNG for reproducability
    cpuThreads = 12

    # Configue CPU/GPU Compute for process
    deviceId = huggingfacehelpers.configure_compute("cpu")

    # Set random seed & CPU threads
    huggingfacehelpers.set_seed_and_cpu_threads(seed = seed, cpuThreads=cpuThreads)

    # Get model location
    fineTunedModelLocationBasePath = r"Models\HappyTransformer-FineTuning-TextGen"
    # Get text generation config
    textGenerationConfig = huggingfacehelpers.TextGenerationConfig()

    # Set this to 1 if running in VSCode, external terminal can be higher
    args = GENEvalArgs(preprocessing_processes=1, mlm_probability=0.2, batch_size=100)

    ########################
    ## EVALUATION         ##
    ########################

    ## Generic models (not fine-tuned)

    # happy_gen = HappyGeneration("GPT2", "EleutherAI/gpt-j-6B")  
    # resultGenericGPTJ6B = happy_gen.eval(evalData, args=args)

    # happy_gen = HappyGeneration()  
    # resultGeneric = happy_gen.eval(evalData, args=args)

    # happy_gen = HappyGeneration("GPT2", "gpt2-xl")
    # resultGenericGPT2XL = happy_gen.eval(evalData, args=args)

    # happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M")  
    # resultGenericGPTNeo125M = happy_gen.eval(evalData, args=args)

    # happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")  
    # resultGenericGPTNeo13B = happy_gen.eval(evalData, args=args)

    happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")  
    resultGenericGPTNeo27B = happy_gen.eval(textGenerationConfig.evalData, args=args)

    ## Fine-tuned models

    # # Fine-Tuned model (GPT2-xl)
    # baseModelArchitecture = "gpt2-xl" # Smaller GPT-Neo model
    # fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

    # if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
    # else :
    #     fineTunedModelLocation = fineTunedModelLocation + "-" + baseModelArchitecture

    # happy_gen = HappyGeneration(model_type="GPT2", model_name=fineTunedModelLocation) 
    # resultGPT2xl = happy_gen.eval(evalData, args=args)

    # Fine-Tuned model (GPT-NEO 125mil)
    # baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller GPT-Neo model
    # fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

    # if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
    # else :
    #     fineTunedModelLocation = fineTunedModelLocation + "-" + baseModelArchitecture

    # happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=fineTunedModelLocation) 
    # resultGPTNeo125m = happy_gen.eval(evalData, args=args)

    # # Fine-Tuned model (GPT-NEO 1.3B)
    # baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Smaller GPT-Neo model
    # fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

    # if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
    # elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    #     fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
    # else :
    #     fineTunedModelLocation = fineTunedModelLocation + baseModelArchitecture

    # happy_gen = HappyGeneration(model_type="GPT-NEO", model_name=fineTunedModelLocation) 
    # resultGPTNeo13B = happy_gen.eval(evalData, args=args)

    # # Fine-Tuned model (GPT-NEO 2.7B)
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
    resultGPTNeo27B = happy_gen.eval(textGenerationConfig.evalData, args=args)

    ########################
    ## RESULTS            ##
    ########################

    # Compare the loss for generic models and the fine-tuned models on the training set (can split for eval)
    print("**** EVAL RESULTS ****")
    # print("Generic GPT2:" + str(resultGeneric.loss))                        # EvalResult(loss=3.216)
    # print("Generic GPT2-XL:" + str(resultGenericGPT2XL.loss))               # EvalResult(loss=2.7335)
    # print("Generic GPT-NEO-125M:" + str(resultGenericGPTNeo125M.loss))      # EvalResult(loss=3.0573)
    #print("Generic GPT-NEO-1.3B:" + str(resultGenericGPTNeo13B.loss))       # EvalResult(loss=2.6203)
    print("Generic GPT-NEO-2.7B:" + str(resultGenericGPTNeo27B.loss))       # EvalResult(loss=2.4492)
    # print("Fine-tuned GPT2-xl:" + str(resultGPT2xl.loss))                   # EvalResult(loss=0.00012)
    #print("Fine-tuned GPT-NEO-125M:" + str(resultGPTNeo125m.loss))          # EvalResult(loss=2.5931)
    # print("Fine-tuned GPT-NEO-1.3B:" + str(resultGPTNeo13B.loss))           # EvalResult(loss=0.0002295)
    print("Fine-tuned GPT-NEO-2.7B:" + str(resultGPTNeo27B.loss))           # EvalResult(loss=0.9876)


# Main entry method
if __name__ == "__main__":
    main()