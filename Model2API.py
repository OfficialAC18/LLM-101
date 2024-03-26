import torch
from peft import PeftModel
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

FT_MODEL = "/scratch/users/k23058970/llama-2-7b-ft-guanaco"
device_map = {'':0}


@app.route("/predict",methods =['POST'] )
def predict():
        prompt = request.json
        print(prompt)
        
        prompt = prompt['prompt']
        
        #Generate the Prompt output
        results = pipe(f"<s>[INST]{prompt}[/INST]")

        #Return the generated content
        return jsonify({'output':results[0]['generated_text'].split().split('[/INST]')[1]})

@app.route('/test')
def test():
    return "Endpoint is working"


if __name__ == '__main__':
    # Load the model
    # model = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path = FT_MODEL,
    #     low_cpu_mem_usage=True,
    #     return_dict = True,
    #     torch_dtype=torch.float16,
    #     device_map = device_map
    # )
    # model = PeftModel.from_pretrained(model,FT_MODEL)
    # model = model.merge_and_unload()

    # #Load the corresponding Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     FT_MODEL,
    #     trust_remote_code = True
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'

    # # Create the pipeline for generation
    # # Search for more arguments laters
    # pipe = pipeline(task="text-generation",
    #                 model = model,
    #                 tokenizer = tokenizer,
    #                 max_length = 1000)

    #Run API
    app.run(debug=True)