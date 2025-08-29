from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_from_subfolder(repo_id, subfolder, device="cuda"):
    """
    Load a model and tokenizer from a specific subfolder in a Hugging Face repository.
    
    Args:
        repo_id: The Hugging Face repository ID (e.g., "meta-llama/Llama-2-7b-hf")
        subfolder: The subfolder path within the repository (e.g., "checkpoint-1000")
        device: Device to load the model on (default: "cuda")
    
    Returns:
        model, tokenizer tuple
    """
    
    # Load tokenizer from subfolder
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        subfolder=subfolder,
        trust_remote_code=True
    )
    
    # Load model from subfolder
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=torch.float16,  # Use fp16 for efficiency
        device_map="auto",  # Automatically distribute model across devices
        trust_remote_code=True
    )
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    # Example: Load a model from a specific checkpoint subfolder
    repo_id = "quanwei0/nq-search-r1-ppo-qwen2.5-7b-em-gae-mixed-reward-new7"
    subfolder = "actor/global_step_500"  # or "step-1000", "epoch-2", etc.
    

    model, tokenizer = load_model_from_subfolder(repo_id, subfolder)
    print(f"Successfully loaded model from {repo_id}/{subfolder}")
    
    # Test the model
    # inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_length=50)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"Model response: {response}")
