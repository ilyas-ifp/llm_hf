from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch

auth_token = "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"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load the pre-trained model and tokenizer
def get_tokenizer_model(name):

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,cache_dir='./model/',token=auth_token,device_map='auto')

    return tokenizer, model


def formate_prompt(name,prompt) :

    pass