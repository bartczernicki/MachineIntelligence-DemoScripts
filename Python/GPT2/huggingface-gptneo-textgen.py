# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer/tree/master/examples/generation



from transformers import pipeline
import torch
import os
import csv
import inspect


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
    deviceId = configure_compute("cpu")

    # Set random seed & CPU threads
    set_seed_and_cpu_threads(seed = seed, cpuThreads=cpuThreads)

    # Get model location
    fineTunedModelLocationBasePath = r"Models\HappyTransformer-FineTuning-TextGen"
    # baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller GPT-Neo model
    # baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger GPT-Neo model
    baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Largest GPT-Neo model
    #baseModelArchitecture = "gpt"
    fineTunedModelLocation = get_finetuned_model_location(baseModelArchitecture, fineTunedModelLocationBasePath)
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
            # Write text generated to CSV
            write_csv_textgenerated(textGenCsv, generatorResults, textGenModel, topK, temperature, topProbabilities)

    # Iterate over generated results list
    # for listItem in generatorResults:
    #     # Access dictionary items
    #     print(str(listItem['generated_text']).rstrip('\n'))


    # for i, textGenResult in enumerate(generatorResults) :
    #     # Extract the sentence
    #     print(str(textGenResult).replace("{'generated_text': '", "").replace("'}", ""))


def configure_compute(desiredCompute):
    # Turn off CUDA (for large fine-tuned modes that won't fit on GPU)
    deviceId = 0

    if (desiredCompute == "cpu") :
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        deviceId = -1
    elif (desiredCompute == "cuda") :
        deviceId = 0
    elif (desiredCompute == "default") :
        deviceId = 0 if torch.cuda.is_available() else -1
    print("CONFIG - device compute: " + str(desiredCompute))

    return deviceId

def set_seed_and_cpu_threads(seed, cpuThreads):
    # set seed for reproducability
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("CONFIG - seed: " + str(seed))

    # Ensure all the CPU threads are being used
    torch.set_num_threads(cpuThreads)
    pyTorchThreads = torch.get_num_threads()
    print("CONFIG - PyTorch config for number of CPU threads: " + str(pyTorchThreads))

def get_finetuned_model_location(baseModelArchitecture, fineTunedModelLocationBasePath = ""):
    modelLocationPath = ""

    if (baseModelArchitecture == r"EleutherAI/gpt-neo-125M") :
        modelLocationPath = fineTunedModelLocationBasePath + "-GPTNeo-125M"
    elif (baseModelArchitecture == r"EleutherAI/gpt-neo-1.3B") :
        modelLocationPath = fineTunedModelLocationBasePath + "-GPTNeo-13B"
    elif (baseModelArchitecture == r"EleutherAI/gpt-neo-2.7B") :
        modelLocationPath = fineTunedModelLocationBasePath + "-GPTNeo-27B"
    else :
        modelLocationPath = fineTunedModelLocationBasePath + baseModelArchitecture

    print("CONFIG - Fine-Tuned Model Location Path: " + str(modelLocationPath))
    return modelLocationPath

def write_csv_textgenerated(textGenCsv, generatorResults, fineTunedModelLocation, topK, temperature, topProbabilities):

    with open(textGenCsv, 'a', encoding='UTF8', newline='') as csvfile: 
        # creating a csv dict writer object 
        writer = csv.writer(csvfile, delimiter = "|") 
    
        # Iterate over generated results list
        for i, textGenResult in enumerate(generatorResults) :
            listToWrite = []
            listToWrite.append("Model=" + fineTunedModelLocation)
            listToWrite.append("Params=top_k:{}tempature:{}top_p:{}".format(topK, temperature, topProbabilities))
            # Access dictionary items
            stringToWrite = str(textGenResult).replace(r"\n\n", " ")
            listToWrite.append(stringToWrite)

            # write the row
            writer.writerow(listToWrite)

# Main entry method
if __name__ == "__main__":
    main()