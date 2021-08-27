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

########################
# FINE TUNING ##
########################

# cwd = os.getcwd()
# print(cwd)
# f = open(r"Data\test.txt", "r")
# print(f.read())

happy_gen = HappyGeneration()
args = GENTrainArgs(num_train_epochs=6) 

happy_gen.train(r"Data\statisticslines.txt", args=args)

# Save the model
happy_gen.save(r"Models\HappyTransformer-FineTuning-TextGen")