import os
from https://huggingface.co/cognitivecomputations/Samantha-1.11-70b/tree/main import hf_hub_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set API key
HUGGING_FACE_API_KEY = os.getenv("HUGGINGFACE_HUB_TOKEN", "hf_KrFoELlKCxWnHDzSHh0N8IjOVPTMsuvPqVH")

# Log in using the token (optional if already set in the environment)
login(token=HUGGING_FACE_API_KEY)

# Define model ID and filenames
model_id = "cognitivecomputations/Samantha-1.11-70b"
filenames = [
    ".gitattributes", "README.md", "config.json", "generation_config.json",
    "pytorch_model-00001-of-00015.bin", "pytorch_model-00002-of-00015.bin",
    "pytorch_model-00003-of-00015.bin", "pytorch_model-00004-of-00015.bin",
    "pytorch_model-00005-of-00015.bin", "pytorch_model-00006-of-00015.bin",
    "pytorch_model-00007-of-00015.bin", "pytorch_model-00008-of-00015.bin",
    "pytorch_model-00009-of-00015.bin", "pytorch_model-00010-of-00015.bin",
    "pytorch_model-00011-of-00015.bin", "pytorch_model-00012-of-00015.bin",
    "pytorch_model-00013-of-00015.bin", "pytorch_model-00014-of-00015.bin",
    "pytorch_model-00015-of-00015.bin", "pytorch_model.bin.index.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json"
]

# Directory to save the downloaded model files
model_dir = "./local_model"
os.makedirs(model_dir, exist_ok=True)

# Download each file
for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=HUGGING_FACE_API_KEY,
        local_dir=model_dir
    )
    print(f"Downloaded {filename} to {downloaded_model_path}")

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Example text generation
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

# Decode and print the result
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
