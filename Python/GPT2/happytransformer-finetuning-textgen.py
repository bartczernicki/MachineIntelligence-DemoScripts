# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer

########################
## SETUP              ##
########################

from happytransformer import HappyGeneration, GENSettings, GENTrainArgs
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
torch.device("cpu")

baseModelType = "GPT2"
baseModelArchitecture = "gpt2"
# baseModelType = "GPT-NEO"
# baseModelArchitecture = "EleutherAI/gpt-neo-125M"

# Load GPT2 medium model
happy_gen = HappyGeneration(baseModelType, baseModelArchitecture)
happy_gen._device = torch.device("cpu")
print(happy_gen._device.type)

# Set up configuration for the model
args = GENTrainArgs(num_train_epochs=50, batch_size=40) 

# # Traid the model
happy_gen.train(r"Data\statisticslines.txt", args=args)

# # Save the model
happy_gen.save(r"Models\HappyTransformer-FineTuning-TextGen")


# ##########################
# ## TEST THE TUNED MODEL ##
# ##########################
happy_gen = HappyGeneration(baseModelType, r"Models\HappyTransformer-FineTuning-TextGen")  # Best performance 

genArgs = GENSettings(max_length=100, no_repeat_ngram_size=2, top_p=0.7, temperature=0.2, early_stopping=True, top_k=300)
result = happy_gen.generate_text("Statistics is ", args=genArgs)
print(result.text)