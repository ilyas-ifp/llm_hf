import argparse
import time

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
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,cache_dir='./model/',token=auth_token,device_map='auto')

    return tokenizer, model

if '__main__' == __name__:

    parser = argparse.ArgumentParser(description='Choose a model to launch.')
    parser.add_argument('model_name', type=str, help='The name of the model to launch')

    args = parser.parse_args()

    print(torch.cuda.is_available())

    name = {
        "Mistral7B": "model/Mistral",
        "Mixtral8x7B": "model/Mixtral",
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

    start = time.time()

    tokenizer, model = get_tokenizer_model(chosen_model)
    print(model.hf_device_map)
    end = time.time()
    time_to_load = (end-start)
    print('Time to Load the model : {:.2f} ms '.format(time_to_load))

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1000,
    )

    prompts = {
        "Mistral7B": """<s>[INST] <<SYS>>
                    You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                    Please ensure that your responses are socially unbiased and positive in nature. 
                    If you lack details or context to answer a question, just say it.
                    <</SYS>> 

                    {question}
                    [/INST]""",
        "Mixtral8x7B": """<s>[INST] <<SYS>>
                    You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                    Please ensure that your responses are socially unbiased and positive in nature. 
                    If you lack details or context to answer a question, just say it.
                    <</SYS>> 

                    {question}
                    [/INST]""",
        "Qwen72B": "Prompt for Qwen72B: {question}",
        "falcon40B": """You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                        Please ensure that your responses are socially unbiased and positive in nature. 
                        If you lack details or context to answer a question, just say it.
                        >>QUESTION<< : {question}
                        >>ANSWER<< """,
        "falcon7B": """You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                        Please ensure that your responses are socially unbiased and positive in nature. 
                        If you lack details or context to answer a question, just say it.
                        >>QUESTION<< : {question}
                        >>ANSWER<< """,
        "Llama27b": """<s>[INST] <<SYS>>
                    You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                    Please ensure that your responses are socially unbiased and positive in nature. 
                    If you lack details or context to answer a question, just say it.
                    <</SYS>> 

                    {question}
                    [/INST]""",
        "Llama213b": """<s>[INST] <<SYS>>
                    You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                    Please ensure that your responses are socially unbiased and positive in nature. 
                    If you lack details or context to answer a question, just say it.
                    <</SYS>> 

                    {question}
                    [/INST]""",
        "Llama270b": """<s>[INST] <<SYS>>
                    You are a helpful assistant Always answer as helpfully as possible, while being safe. Your answers should not includeany harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
                    Please ensure that your responses are socially unbiased and positive in nature. 
                    If you lack details or context to answer a question, just say it.
                    <</SYS>> 

                    {question}
                    [/INST]""",
        "phi2": """Prompt for phi2: {question}"""
    }

    prompt_template = prompts.get(args.model_name, "Default prompt: {question}")

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    llm = HuggingFacePipeline(pipeline=pipeline) 
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

    while True:
        user_input = input("You (write quit to end the conversation) : ")
        if user_input.lower() == 'quit':
            break
        begin_generation = time.time()
        response = chain.run(user_input)
        end_generation = time.time()
        print("AI : ", response)
        print("Inference Time : {:.2f} ms ".format(end_generation-begin_generation))

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