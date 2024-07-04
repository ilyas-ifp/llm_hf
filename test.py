from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Environment variables
auth_token = "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"


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load the pre-trained model and tokenizer
def get_tokenizer_model(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/',
                                                  use_auth_token=auth_token, trust_remote_code=True, device_map="cuda") 

    return tokenizer, model


if '__main__' == __name__:

    print(torch.cuda.is_available())

    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer, model = get_tokenizer_model(name)

    print(model)

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=300,
    )

    prompt_template = """[INST] {question} [/INST] """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    llm = HuggingFacePipeline(pipeline=pipeline) 
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

    response = chain.run("say : it works, in French")

    print(response)

    
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Iterate over all available GPUs
        for i in range(torch.cuda.device_count()):
            # Select the GPU
            torch.cuda.set_device(i)
            # Get the name of the current GPU
            gpu_name = torch.cuda.get_device_name(i)
            # Get the total memory of the current GPU
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # Convert total memory from bytes to GB
            total_memory_gb = total_memory / (1024 ** 3)
            # Get the current memory usage
            current_memory_allocated = torch.cuda.memory_allocated(i)
            # Convert current memory usage to GB
            current_memory_allocated_gb = current_memory_allocated / (1024 ** 3)
            # Get the current memory cached
            current_memory_cached = torch.cuda.memory_reserved(i)
            # Convert current memory cached to GB
            current_memory_cached_gb = current_memory_cached / (1024 ** 3)

            print(f"GPU {i} ({gpu_name}):")
            print(f"  Total Memory: {total_memory_gb:.2f} GB")
            print(f"  Currently Allocated Memory: {current_memory_allocated_gb:.2f} GB")
            print(f"  Currently Cached Memory: {current_memory_cached_gb:.2f} GB")
    else:
        print("CUDA is not available.")