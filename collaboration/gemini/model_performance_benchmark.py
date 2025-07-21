import time
import subprocess
import json
import argparse

# --- Models to Benchmark ---
# Add the models you want to test here
MODELS_TO_BENCHMARK = [
    "deepseek-coder:33b",
    "mistral:7b",
    "llama3.2:3b",
    "deepseek-r1:8b",
    "llama3.2:1b",
]

# --- Benchmark Prompts ---
# A dictionary of prompts to test, categorized by task type
BENCHMARK_PROMPTS = {
    "code_generation": "Write a Python function to calculate the factorial of a number.",
    "summarization": "Summarize the following text: [Insert a long text here]",
    "general_q&a": "What is the capital of France?",
    "reasoning": "If all men are mortal, and Socrates is a man, what can you conclude about Socrates?",
}

def run_benchmark(model_name, prompt):
    """Runs a single benchmark test for a given model and prompt."""
    print(f"  Testing prompt: \"{prompt[:50]}...\"")
    start_time = time.time()

    try:
        # Use 'ollama run' to get a response from the model
        process = subprocess.run(
            ["ollama", "run", model_name, prompt],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5-minute timeout for the model response
        )
        end_time = time.time()
        duration = end_time - start_time

        return {
            "status": "success",
            "duration_seconds": round(duration, 2),
            "response_length": len(process.stdout.strip()),
        }

    except subprocess.TimeoutExpired:
        print(f"    ❌ Timeout expired for model {model_name}.")
        return {"status": "error", "error_message": "Timeout expired"}
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Error running model {model_name}: {e.stderr}")
        return {"status": "error", "error_message": e.stderr}
    except FileNotFoundError:
        print("❌ 'ollama' command not found. Is Ollama installed and in your PATH?")
        exit(1)

def main():
    """Main function to run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark the performance of Ollama models.")
    parser.add_argument("--output", default="benchmark_results.json", help="File to save the benchmark results.")
    args = parser.parse_args()

    print("--- Starting Model Performance Benchmark ---")
    benchmark_results = {}

    for model in MODELS_TO_BENCHMARK:
        print(f"\n--- Benchmarking Model: {model} ---")
        benchmark_results[model] = {}

        for task, prompt in BENCHMARK_PROMPTS.items():
            print(f"  Task: {task}")
            result = run_benchmark(model, prompt)
            benchmark_results[model][task] = result

    # Save results to a JSON file
    with open(args.output, "w") as f:
        json.dump(benchmark_results, f, indent=4)

    print(f"\n--- Benchmark Complete! ---")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
