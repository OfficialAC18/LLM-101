#Link: https://github.com/krishnaik06/Complete-Langchain-Tutorials/blob/main/Blog%20Generation/app.py
import os
os.environ['HF_HOME'] = './hfhub'
import torch
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig


def model_load():
    #Load the LLaMa Model and Tokenizer
    MODEL = "OfficialAC18/llama-2-7b-ft-guanaco"
    device_map = {'':0}

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = MODEL,
        low_cpu_mem_usage=True,
        return_dict = True,
        torch_dtype=torch.float16,
        device_map = device_map
    )

    #Reload Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL,
                                            trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.paddding_side = "right"

    #Add some decoding strategies
    generation_config = GenerationConfig(
        num_beams = 4,
        early_stopping = True,
        eos_token_id = model.generation_config.eos_token_id,
        bos_token_id = model.generation_config.bos_token_id,
        pad_token_id = model.generation_config.pad_token_id,
        num_return_sequences = 1,
    )

    model.generation_config = generation_config

    #Creating Pipeline
    pipe = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_length = 300
    )

    #Wrap around LangChain Pipeline
    llm = HuggingFacePipeline(pipeline = pipe)
    return llm



## Function To get response from LLaMa 2 model
def getLLamaresponse(task,no_words,load_model = False):   
    ## Prompt Template
    if load_model:
        llm = model_load()

    template="""
        Generate Python Code for {task} using at most {no_words} words,
        do not give any extra information, do not repeat yourself and put ```
        at the start and end of the code block.
            """
    
    prompt=PromptTemplate(input_variables=["task",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(task=task,no_words=no_words))
    print(response)
    return response






st.set_page_config(page_title="Python Code Generator",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Python Code Generator 🤖")

task=st.text_input("Write Query Here")

## creating two more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.selectbox('Length of code',
                            ('150','200','250'),index=0)
    
submit=st.button("Generate")
count = 0
## Final response
if submit:
    if count == 0:
        st.write(getLLamaresponse(task,no_words,load_model = True))
    else:
        st.write(getLLamaresponse(task,no_words))
    
    count +=1