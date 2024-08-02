from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# Global constants for the model
MODEL_PATH = "model/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
CONTEXT_SIZE = 2048

# Load parameters
model_params = {
    "n_batch": 512,
    "rope_freq_base": 0,
    "rope_freq_scale": 0,
    "n_gpu_layers": 0,
    "use_mlock": True,
    "main_gpu": 0,
    "tensor_split": [0],
    "seed": -1,
    "f16_kv": True,
    "use_mmap": True,
    "no_kv_offload": False,
    "num_experts_used": 0
}

inference_params = {
    "n_threads": 4,
    "n_predict": -1,
    "top_k": 40,
    "min_p": 0.05,
    "top_p": 0.95,
    "temp": 0.2,
    "repeat_penalty": 1.1,
    "input_prefix": "user\n\n",
    "input_suffix": "assistant\n\n",
    "antiprompt": ["", ""],
    "pre_prompt": "",
    "pre_prompt_suffix": "",
    "pre_prompt_prefix": "system\n\n",
    "seed": -1,
    "tfs_z": 1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n_keep": 0,
    "logit_bias": {},
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "memory_f16": True,
    "multiline_input": False,
    "penalize_nl": True
}

# Load the Llama model
zephyr_model = Llama(model_path=MODEL_PATH, n_ctx= CONTEXT_SIZE,verbose=False, chat_format="llama-3")


@app.route('/v1/chat/completions', methods=['POST'])
def handle_completions():
    # Check for valid authorization token
    auth_token = request.headers.get('Authorization')
    if auth_token != "Bearer lm-studio":
        return jsonify({"error": "Unauthorized"}), 401

    # Extract data from POST request
    data = request.json
    user_prompt = data.get('messages')[0]['content'] if data.get('messages') else ''
    temperature = data.get('temperature', 0.2)


    try:
        # Await the model to generate the response
        model_output = zephyr_model.create_chat_completion(
            messages= [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ], temperature=temperature
    
        )
        print(model_output)
        # Respond with the generated text
        return model_output, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=1234)
