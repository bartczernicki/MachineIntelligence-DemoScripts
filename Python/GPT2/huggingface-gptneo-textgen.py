# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer/tree/master/examples/generation


import huggingfacehelpers # Custom module helpers
from transformers import pipeline # Huggingface transformers
import time

def main():

    ########################
    ## CONFIG             ##
    ########################

    # Config Variables
    seed = 100
    cpuThreads = 12
    sentencesStartForTextGeneration = ['Statistics can be used to help make decisions.', 'Data science is used in sports.', 'Baseball coaches use statistics for ',
        'Making decisions can be aided by probabilistic approaches.', 'Sports analytics includes using ', 'There are many ways to use statistics in sports.',
        'Machine intelligence can help the decision making process', 'A decision support system is ']

    # Configue CPU/GPU Compute for process
    deviceId = huggingfacehelpers.configure_compute("cpu")

    # Set random seed & CPU threads
    huggingfacehelpers.set_seed_and_cpu_threads(seed = seed, cpuThreads=cpuThreads)

    # Get model location
    fineTunedModelLocationBasePath = r"Models\HappyTransformer-FineTuning-TextGen"
    baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller GPT-Neo model

    # baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger GPT-Neo model
    #baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Largest GPT-Neo model
    #baseModelArchitecture = "gpt"
    fineTunedModelLocation = huggingfacehelpers.get_finetuned_model_location(baseModelArchitecture, fineTunedModelLocationBasePath)
    modelsForTextGeneration = []
    modelsForTextGeneration.append(baseModelArchitecture)
    modelsForTextGeneration.append(fineTunedModelLocation)

    # Data Location (format appopriate for OS)
    textGenCsv = r"Data\TextGeneratedFromModels.txt"

    # Parameters for text-generation
    maxLength = 120
    topK = 100
    temperature = 0.5
    topProbabilities = 0.92
    numberOfSentenceSequences = 100
    noRepeatNgramSize = 3

    ################################
    ## TEXT GENARATION            ##
    ################################


    for textGenModel in modelsForTextGeneration:
        # Load the generator once for each model pass
        generator = pipeline(task="text-generation", model=textGenModel, device=deviceId, framework="pt", use_fast=False)

        for sentenceStart in sentencesStartForTextGeneration:
            print("Performing text generation using: {}. Sentence: {}".format(textGenModel, sentenceStart))
            startTime = time.time()

            # Generate text on (generic) pre-trained model
            generatorResults = generator(
                sentenceStart,
                clean_up_tokenization_spaces = True,
                do_sample=True, 
                max_length=maxLength,
                top_k=topK,
                temperature=temperature,
                top_p=topProbabilities,
                no_repeat_ngram_size=noRepeatNgramSize,
                num_return_sequences=numberOfSentenceSequences
            )

            # Print elapsed time
            endTime = time.time()
            print("Time elapsed generating text: ", endTime - startTime)

            # Write text generated to CSV
            # huggingfacehelpers.write_csv_textgenerated(textGenCsv, generatorResults, textGenModel, topK, temperature, topProbabilities)

            # Debug Iterate over generated results list
            # for listItem in generatorResults:
            #     # Access dictionary items
            #     print(str(listItem['generated_text']).rstrip('\n'))


# Main entry method
if __name__ == "__main__":
    main()