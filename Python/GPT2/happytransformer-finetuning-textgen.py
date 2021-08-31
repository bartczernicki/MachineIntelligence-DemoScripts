# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer

########################
## SETUP              ##
########################

from happytransformer import HappyGeneration, GENSettings, GENTrainArgs
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ensure all the CPU threads are being used
torch.set_num_threads(12)
pyTorchThreads = torch.get_num_threads()
print("PyTorch number of CPU threads: " + str(pyTorchThreads))

# happy_gen = HappyGeneration("GPT2", "gpt2-xl")  # Best performance 

# args = GENSettings(max_length=75, no_repeat_ngram_size=3, top_p=0.94, temperature=0.7)
# result = happy_gen.generate_text("You should learn statistics. ", args=args)
# print(result.text)

# cwd = os.getcwd()
# print(cwd)
# f = open(r"Data\test.txt", "r")
# print(f.read())


#########################
## FINE TUNING MODEL   ##
#########################
torch.cuda.empty_cache()
torch.manual_seed(255)

baseModelType = "GPT2"

#baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller model
#baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger model
baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Larger model
#baseModelArchitecture = "gpt2-xl"

fineTunedModelLocation = r"Models\HappyTransformer-FineTuning-TextGen"

if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-125M"
    baseModelType = "GPT-NEO"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-13B"
    baseModelType = "GPT-NEO"
elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
    fineTunedModelLocation = fineTunedModelLocation + "-GPTNeo-27B"
    baseModelType = "GPT-NEO"
else :
    fineTunedModelLocation = fineTunedModelLocation + "-" + baseModelArchitecture
    baseModelType = "GPT2"


# Load model type and the architecture/pre-trained model
happy_gen = HappyGeneration(baseModelType, baseModelArchitecture)

# Set up configuration for the model
args = GENTrainArgs(num_train_epochs=75, batch_size=102) 

# # Traid the model
happy_gen.train(r"Data\statisticslines.txt", args=args)

# # Save the model
happy_gen.save(fineTunedModelLocation)


# ##########################
# ## TEST THE TUNED MODEL ##
# ##########################
happy_gen = HappyGeneration(baseModelType, fineTunedModelLocation)  # Best performance 

genArgs = GENSettings(max_length=100, no_repeat_ngram_size=2, top_p=0.92, temperature=0.6, early_stopping=True, top_k=400)
result = happy_gen.generate_text("You should learn statistics ", args=genArgs)
print(result.text)