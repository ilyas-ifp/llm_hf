import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline


auth_token = "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"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load the pre-trained model and tokenizer
def get_tokenizer_model(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name,cache_dir='./model_cache', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,cache_dir='./model_cache',
                                                  use_auth_token=auth_token, trust_remote_code=True,quantization_config=quantization_config ,device_map="auto") 


    tokenizer.save_pretrained("./model/Llama2_7B")
    model.save_pretrained("./model/Llama2_7B")
    print('saved')
    return tokenizer, model
                                                                     
if '__main__' == __name__:

    parser = argparse.ArgumentParser(description='Choose a model to launch.')
    parser.add_argument('model_name', type=str, help='The name of the model to launch')

    args = parser.parse_args()

    print(torch.cuda.is_available())

    name = {
        "Mistral7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "Mixtral8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Qwen72B": "Qwen/Qwen-72B",
        "falcon40B": "tiiuae/falcon-40b-instruct",
        "falcon7B": "tiiuae/falcon-7b-instruct",
        "Llama27b": "meta-llama/Llama-2-7b-chat-hf",
        "Llama213b": "meta-llama/Llama-2-13b-chat-hf",
        "Llama270b": "meta-llama/Llama-2-70b-chat-hf",
        "phi2": "microsoft/phi-2"
    }

    chosen_model = name.get(args.model_name, "Model not found")

    if chosen_model == "Model not found":
        print("Invalid model name. Please try again.")
        exit()

    tokenizer, model = get_tokenizer_model(chosen_model)
    print(model)