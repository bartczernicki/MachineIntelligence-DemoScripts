# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer

########################
## SETUP              ##
########################

from happytransformer import HappyGeneration, GENSettings, GENTrainArgs
import os

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

# Load GPT@ large model
happy_gen = HappyGeneration("GPT2", "gpt2-large")

# # Set up configuration for the model
# args = GENTrainArgs(num_train_epochs=6) 

# # # Traid the model
# happy_gen.train(r"Data\statisticslines.txt", args=args)

# # # Save the model
# happy_gen.save(r"Models\HappyTransformer-FineTuning-TextGen")


##########################
## TEST THE TUNED MODEL ##
##########################
happy_gen = HappyGeneration("GPT2", r"Models\HappyTransformer-FineTuning-TextGen")  # Best performance 

genArgs = GENSettings(max_length=100, no_repeat_ngram_size=2, top_p=0.7, temperature=0.2, early_stopping=True, top_k=300)
result = happy_gen.generate_text("Statistics is ", args=genArgs)
print(result.text)