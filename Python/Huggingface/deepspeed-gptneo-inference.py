import huggingfacehelpers # Custom module helpers
from transformers import GPTNeoForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import deepspeed
import torch

#baseModelArchitecture = "gpt2" 
#baseModelArchitecture = "EleutherAI/gpt-neo-125M" # Smaller model
#baseModelArchitecture = "EleutherAI/gpt-neo-1.3B" # Larger model
baseModelArchitecture = "EleutherAI/gpt-neo-2.7B" # Larger model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("GPT2 Model Device Type: " + device)
#gpt2Model = r"Models\HappyTransformer-FineTuning-TextGen" #fine-tuned model
torch.manual_seed(100)
torch.cuda.manual_seed(100)

textGenerationConfig = huggingfacehelpers.TextGenerationConfig()
# Get model location
fineTunedModelLocationBasePath = r"Models\HappyTransformer-FineTuning-TextGen"

fineTunedModelLocation = huggingfacehelpers.get_finetuned_model_location(baseModelArchitecture, fineTunedModelLocationBasePath)
fineTunedModelLocation = baseModelArchitecture
print(fineTunedModelLocation)
# Sentence to start text-generation
sentenceStart = 'You should learn statistics'

# casting to fp16 "half" gives a large speedup during model loading
# model = GPTNeoForCausalLM.from_pretrained(baseModelArchitecture).half().to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(baseModelArchitecture)

# Load gpt2 model, can be gpt-large, gpt-xl or custom etc.
tokenizer = GPT2Tokenizer.from_pretrained(fineTunedModelLocation)

# Add the EOS token as PAD token to avoid warnings, send to proper compute device
model = GPTNeoForCausalLM.from_pretrained(fineTunedModelLocation, pad_token_id=tokenizer.eos_token_id)
model.half().to(device)

# ds_engine = deepspeed.init_inference(model,
#                                  mp_size=2,
#                                  dtype=torch.half,
#                                  checkpoint=None,
#                                  replace_method='auto')

# Encode context the generation is conditioned on, send to proper compute device

input_ids = tokenizer.encode(sentenceStart, return_tensors='pt')
input_ids_OnDevice = input_ids.to(device)


########################
## TEXT GENERATION    ##
########################

## 1) TEXT GENERATION - GPT2 - GREEDY
# Generate text until the output length (which includes the context length) reaches 75
greedy_output = model.generate(input_ids_OnDevice, max_length=75)

samplingProbability_outputs = model.generate(input_ids_OnDevice, 
    do_sample=True, 
    max_length=75,
    # Mix of outputs
    top_k=400,
    temperature=0.75,
    top_p=0.9, 
    # Multiple sentences returned
    num_return_sequences=10
)

# Notice the text-gen output repeats itself
print("Greedy Output:\n" + 100 * '-')
print(tokenizer.decode(samplingProbability_outputs[0], skip_special_tokens=True))
print(tokenizer.decode(samplingProbability_outputs[1], skip_special_tokens=True))
print(tokenizer.decode(samplingProbability_outputs[2], skip_special_tokens=True))
print(tokenizer.decode(samplingProbability_outputs[3], skip_special_tokens=True))
print(tokenizer.decode(samplingProbability_outputs[4], skip_special_tokens=True))


