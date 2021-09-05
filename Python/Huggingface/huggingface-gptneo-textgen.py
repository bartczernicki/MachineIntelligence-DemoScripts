# https://happytransformer.com/text-generation/
# https://github.com/EricFillion/happy-transformer/tree/master/examples/generation


import huggingfacehelpers # Custom module helpers
from transformers import pipeline, GPTNeoForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel # Huggingface transformers
import time
import random

def main():

    ########################
    ## CONFIG             ##
    ########################

    # Config Variables
    seed = random.randint(1, 100000) # Set to static value instead of RNG for reproducability
    cpuThreads = 6
    sentencesStartForTextGeneration = ['Statistics can be used to help make decisions.', 'Data science is used in sports.', 'Baseball coaches use statistics for',
        'Making decisions can be aided by probabilistic approaches.', 'Sports analytics includes using ', 'There are many ways to use statistics in sports.',
        'Machine intelligence can help the decision making process', 'A decision support system is']
    numberOfIterations = 5

    # Configue CPU/GPU Compute for process
    deviceName = 'cuda' if huggingfacehelpers.torch.cuda.is_available() else 'cpu'
    # deviceName = 'cpu' # Can set expliticy, uncomment to force CPU
    
    deviceId = huggingfacehelpers.configure_compute(deviceName)

    # Set random seed & CPU threads
    huggingfacehelpers.set_seed_and_cpu_threads(seed = seed, cpuThreads=cpuThreads)

    # Get model location
    fineTunedModelLocationBasePath = r"Models\HappyTransformer-FineTuning-TextGen"
    # Get text generation config
    textGenerationConfig = huggingfacehelpers.TextGenerationConfig()

    fineTunedModelLocation = huggingfacehelpers.get_finetuned_model_location(textGenerationConfig.baseModelArchitecture, fineTunedModelLocationBasePath)
    modelsForTextGeneration = []
    # Add both fine-tuned model and the baseline (generic model)
    modelsForTextGeneration.append(fineTunedModelLocation)
    modelsForTextGeneration.append(textGenerationConfig.baseModelArchitecture)


    # Data Location (format appopriate for OS)
    textGenCsv = r"Data\TextGeneratedFromModels.csv"



    ################################
    ## TEXT GENARATION            ##
    ################################

    for iteration in range(numberOfIterations):

        # Get text generation config (random text-gen parameters)
        textGenerationConfig = huggingfacehelpers.TextGenerationConfig(generateRandom=True)


        for textGenModel in modelsForTextGeneration:

            if (deviceName == "cpu"):
                # CPU PIPELINE FOR TEXT GENERATION

                # Load the generator once for each model pass
                generator = pipeline(task="text-generation", model=textGenModel, device=deviceId, framework="pt", use_fast=False)


                for sentenceStart in sentencesStartForTextGeneration:
                    print("Performing text generation using: {}. Sentence: {}".format(textGenModel, sentenceStart))
                    startTime = time.time()

                    # Generate text on (generic) pre-trained model
                    generatorResults = generator(
                        sentenceStart,
                        clean_up_tokenization_spaces = textGenerationConfig.clean_up_tokenization_spaces,
                        do_sample=textGenerationConfig.do_sample,
                        min_length=textGenerationConfig.min_length,
                        max_length=textGenerationConfig.max_length,
                        top_k=textGenerationConfig.top_k,
                        temperature=textGenerationConfig.temperature,
                        top_p=textGenerationConfig.top_p,
                        no_repeat_ngram_size=textGenerationConfig.no_repeat_ngram_size,
                        num_return_sequences=textGenerationConfig.num_return_sequences
                    )

                    # Print elapsed time
                    timeElapsed = round(time.time() - startTime, 2)
                    print("Time elapsed generating text: ", timeElapsed)

                    #Write text generated to CSV
                    huggingfacehelpers.write_csv_textgenerated(textGenCsv, generatorResults, textGenModel, timeElapsed, seed,
                        textGenerationConfig.top_k, textGenerationConfig.temperature, textGenerationConfig.top_p, textGenerationConfig.no_repeat_ngram_size)

                    # Debug Iterate over generated results list
                    # for listItem in generatorResults:
                    #     # Access dictionary items
                    #     print(str(listItem['generated_text']).rstrip('\n'))

            elif (deviceName == "cuda"):
                # GPU (CUDA) PIPELINE FOR TEXT GENERATION

                # Load the tokenizer once from persisted storage location once and place it into GPU memory
                tokenizer = GPT2Tokenizer.from_pretrained(fineTunedModelLocation)

                # Add the EOS token as PAD token to avoid warnings, send to proper compute device
                # Use FP16 precision vs FP32 to put entire large models into memory
                model = GPTNeoForCausalLM.from_pretrained(fineTunedModelLocation, pad_token_id=tokenizer.eos_token_id)
                model.half().to(deviceName)

                for sentenceStart in sentencesStartForTextGeneration:
                    print("Performing text generation using: {}. Sentence: {}".format(textGenModel, sentenceStart))
                    startTime = time.time()

                    input_ids = tokenizer.encode(sentenceStart, return_tensors='pt')
                    input_ids_OnDevice = input_ids.to(deviceName)

                    generatorResultsTokens = model.generate(input_ids_OnDevice,
                        return_full_text=False,
                        clean_up_tokenization_spaces = textGenerationConfig.clean_up_tokenization_spaces,
                        do_sample=textGenerationConfig.do_sample,
                        min_length=textGenerationConfig.min_length,
                        max_length=textGenerationConfig.max_length,
                        top_k=textGenerationConfig.top_k,
                        temperature=textGenerationConfig.temperature,
                        top_p=textGenerationConfig.top_p,
                        no_repeat_ngram_size=textGenerationConfig.no_repeat_ngram_size,
                        num_return_sequences=textGenerationConfig.num_return_sequences
                    )

                    # Print elapsed time
                    timeElapsed = round(time.time() - startTime, 2)
                    print("Time elapsed generating text: ", timeElapsed)

                    # Decode the tokens into sentences
                    generatorResults = []
                    for generatedToken in generatorResultsTokens:
                        sentence = tokenizer.decode(generatedToken, skip_special_tokens=True)
                        generatorResults.append(sentence)

                    # Write text generated to CSV
                    huggingfacehelpers.write_csv_textgenerated(textGenCsv, generatorResults, textGenModel, timeElapsed, seed,
                        textGenerationConfig.top_k, textGenerationConfig.temperature, textGenerationConfig.top_p, textGenerationConfig.no_repeat_ngram_size)

# Main entry method
if __name__ == "__main__":
    main()