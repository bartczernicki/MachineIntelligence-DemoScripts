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

# baseModelType = "GPT2"
# baseModelArchitecture = "gpt2-large"
baseModelType = "GPT-NEO"
baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller model
baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger model

# Load GPT2 medium model
happy_gen = HappyGeneration(baseModelType, baseModelArchitecture)

# Set up configuration for the model
args = GENTrainArgs(num_train_epochs=25, batch_size=40) 

# # Traid the model
happy_gen.train(r"Data\statisticslines.txt", args=args)

# # Save the model
happy_gen.save(r"Models\HappyTransformer-FineTuning-TextGen")


# ##########################
# ## TEST THE TUNED MODEL ##
# ##########################
happy_gen = HappyGeneration(baseModelType, r"Models\HappyTransformer-FineTuning-TextGen")  # Best performance 

genArgs = GENSettings(max_length=100, no_repeat_ngram_size=2, top_p=0.9, temperature=0.75, early_stopping=True, top_k=300)
result = happy_gen.generate_text("You should learn statistics ", args=genArgs)
print(result.text)