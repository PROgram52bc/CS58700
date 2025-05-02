# Text Generation with RoPE GPT Model

This project supports autoregressive text generation using a trained GPT model with rotary positional embeddings.

## Usage

Run the following command to generate text:

python generate.py --temperature 1 --top_k 5 --prompt "The ship was"

### Options

--prompt: Starting text prompt for generation (default: "The ship was")  
--model_path: Path to the trained model checkpoint (default: models/best_gpt_model.pt)  
--max_tokens: Maximum number of tokens to generate (default: 1000)  
--temperature: Sampling temperature (default: 0.8)  
--top_k: Limits sampling to the top-k most probable tokens (default: 40)  
--use_kv_cache: Use KV caching for efficient autoregressive generation

## Example

python generate.py --temperature 0.9 --top_k 10 --prompt "In the distant future," --use_kv_cache
