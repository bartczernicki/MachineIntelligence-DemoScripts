import torch
import os
import csv


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

def write_csv_textgenerated(textGenCsv, generatorResults, fineTunedModelLocation, timeElapsed, topK, temperature, topProbabilities, noRepeatNgramSize):

    with open(textGenCsv, 'a', encoding='UTF8', newline='') as csvfile: 
        # creating a csv dict writer object 
        writer = csv.writer(csvfile, delimiter = "|") 
    
        # Iterate over generated results list
        for i, textGenResult in enumerate(generatorResults) :
            listToWrite = []
            listToWrite.append("Model=" + fineTunedModelLocation)
            listToWrite.append("TimeElapsed=" + str(timeElapsed))
            listToWrite.append("Params=top_k:{}tempature:{}top_p:{}noRepeatNgramSize:{}".format(topK, temperature, topProbabilities, noRepeatNgramSize))
            # Access dictionary items
            stringToWrite = str(textGenResult).replace(r"\n\n", " ")
            listToWrite.append(stringToWrite)

            # write the row
            writer.writerow(listToWrite)

class TextGenerationConfig:

    def __init__(self):
        #self.baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger GPT-Neo model
        self.baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Largest GPT-Neo model
        #self.baseModelArchitecture = "gpt"
        #self.baseModelArchitecture = str(r"EleutherAI/gpt-neo-125M") # Smaller GPT-Neo model
        self.clean_up_tokenization_spaces = True
        self.do_sample = True
        self.min_length = 100
        self.max_length = 150
        self.top_k = 160
        self.temperature = 0.4
        self.top_p = 0.86
        self.no_repeat_ngram_size = 4
        self.num_return_sequences = 50