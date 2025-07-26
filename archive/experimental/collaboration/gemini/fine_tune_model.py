import os
import json
import subprocess

# --- Configuration ---
MODEL_NAME = "deepseek-coder:33b"
DATASET_PATH = "path/to/your/dataset.jsonl"  # <-- IMPORTANT: Update this path
TUNED_MODEL_NAME = "tuned-deepseek-coder:33b"
OLLAMA_API_URL = "http://localhost:11434"

def check_ollama_connection():
    """Checks if the Ollama server is running."""
    try:
        subprocess.run(["ollama", "list"], check=True, capture_output=True)
        print("✅ Ollama server is running.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama server not found or not running.")
        print("   Please start the Ollama server to proceed.")
        return False

def pull_base_model():
    """Pulls the base model from the Ollama registry if it doesn't exist."""
    print(f"--- Checking for base model: {MODEL_NAME} ---")
    try:
        # Check if the model exists locally
        result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True)
        if MODEL_NAME in result.stdout:
            print(f"✅ Base model '{MODEL_NAME}' already exists locally.")
            return

        # Pull the model if it doesn't exist
        print(f"Pulling base model: {MODEL_NAME}...")
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        print(f"✅ Successfully pulled {MODEL_NAME}.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error pulling model: {e}")
        exit(1)
    except FileNotFoundError:
        print("❌ 'ollama' command not found. Is Ollama installed and in your PATH?")
        exit(1)

def create_modelfile(dataset_path):
    """Creates a Modelfile for fine-tuning."""
    modelfile_content = f"""
FROM {MODEL_NAME}

# --- System Message ---
# Set a custom system message to guide the model's behavior
SYSTEM """You are a specialized coding assistant fine-tuned on a custom dataset.
Your purpose is to generate high-quality, efficient, and clean code based on user prompts.
Adhere to the coding style and patterns found in the fine-tuning data."""

# --- Adapter ---
# The adapter is where the fine-tuning data is applied.
# Ollama uses a file path for the adapter.
ADAPTER "{os.path.abspath(dataset_path)}"
"""
    modelfile_path = "Modelfile.tuned"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print(f"✅ Created Modelfile: {modelfile_path}")
    return modelfile_path

def run_fine_tuning(modelfile_path):
    """Runs the fine-tuning process using ollama create."""
    print(f"--- Starting fine-tuning for: {TUNED_MODEL_NAME} ---")
    print("This process may take a significant amount of time and resources.")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset file not found at '{DATASET_PATH}'")
        print("   Please provide a valid dataset file in JSONL format.")
        print("   Each line should be a JSON object with 'prompt' and 'completion' keys.")
        return

    try:
        # The 'ollama create' command builds the new model from the Modelfile
        subprocess.run(
            ["ollama", "create", TUNED_MODEL_NAME, "-f", modelfile_path],
            check=True
        )
        print(f"✅ Successfully created fine-tuned model: {TUNED_MODEL_NAME}")
        print("--- Fine-tuning complete! ---")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during fine-tuning: {e}")
    except FileNotFoundError:
        print("❌ 'ollama' command not found. Is Ollama installed and in your PATH?")

def main():
    """Main function to run the fine-tuning pipeline."""
    print("--- SutazAI Model Fine-Tuning Script ---")

    if not check_ollama_connection():
        return

    pull_base_model()
    modelfile_path = create_modelfile(DATASET_PATH)
    run_fine_tuning(modelfile_path)

    print("\nTo use your new model, run:")
    print(f"  ollama run {TUNED_MODEL_NAME}")

if __name__ == "__main__":
    main()
