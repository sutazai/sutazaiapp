import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class CPUOptimizedTransformer:
    def __init__(self, model_path: str, quantize_bits: int = 8):
        """Initialize a CPU-optimized transformer model with quantization.

        Args:
            model_path: Path to the model weights
            quantize_bits: Model quantization bit depth (4 or 8)
        """
        self.model_path = model_path
        self.quantize_bits = quantize_bits
        self.model = None
        self.tokenizer = None
        self.thread_pool = ThreadPoolExecutor(
            max_workers=12
        )  # E5-2640 has 12 physical cores
        self._load_model()

    def _load_model(self):
        """Load the model with appropriate quantization."""
        # Set thread count for optimal performance
        torch.set_num_threads(12)  # E5-2640 has 12 physical cores

        # Configure loading parameters
        load_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

        # Apply quantization based on specified bit depth
        if self.quantize_bits == 8:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.quantize_bits == 4:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            load_kwargs["bnb_4bit_use_double_quant"] = True
            load_kwargs["bnb_4bit_quant_type"] = "nf4"
            load_kwargs["device_map"] = "auto"
        else:
            # FP16 fallback with CPU mapping
            load_kwargs["device_map"] = "cpu"

        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )

        # Apply BetterTransformer optimization if available and not quantized
        if self.quantize_bits == 0:
            try:
                self.model = self.model.to_bettertransformer()
                print("Applied BetterTransformer optimization")
            except Exception as e:
                print(f"BetterTransformer optimization not available: {e}")

        # Load tokenizer (lightweight)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set model to evaluation mode
        self.model.eval()

    def _determine_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on CPU cache and memory.

        Returns:
            Optimal batch size for the E5-2640 processor
        """
        # E5-2640 has 15MB L3 cache per CPU, so 30MB total for dual-socket
        # We need to ensure batch fits in L3 cache for optimal performance

        # Check if model is loaded
        if self.model is None:
            return 1

        # Check memory usage of model config
        try:
            model_hidden_size = getattr(self.model.config, "hidden_size", 768)

            # Each token in a batch uses approximately hidden_size*4 bytes (float32)
            # Or hidden_size*2 bytes for float16
            # Target to use at most 1/3 of L3 cache
            bytes_per_token = model_hidden_size * 2  # Assuming fp16

            # Calculate batch size that fits in ~10MB (1/3 of total L3)
            cache_optimal_batch = int(10 * 1024 * 1024 / bytes_per_token)

            # Cap batch size based on CPU core count
            cores = os.cpu_count() or 12
            core_optimal_batch = max(
                1, cores // 4
            )  # 1 batch per 4 cores is a good heuristic

            # Use the smaller of the two values
            return min(cache_optimal_batch, core_optimal_batch)
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            # Fall back to a conservative default if calculation fails
            return 1

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> str:
        """Generate text with optimized CPU usage

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response
        """
        # Determine optimal batch size for CPU
        batch_size = self._determine_optimal_batch_size()

        # Submit to thread pool to avoid blocking
        future = self.thread_pool.submit(
            self._generate_with_batching,
            prompt,
            max_tokens,
            batch_size,
            temperature,
            top_p,
        )
        return future.result()

    def _generate_with_batching(
        self,
        prompt: str,
        max_tokens: int,
        batch_size: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Internal method to perform generation with batching

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            batch_size: Batch size for efficient CPU usage
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate with chunking for memory efficiency
        generated_tokens = []
        chunk_size = min(max_tokens, 256)  # Process in smaller chunks to avoid OOM
        remaining = max_tokens

        while remaining > 0:
            current_chunk = min(chunk_size, remaining)

            # Generate current chunk
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=current_chunk,
                do_sample=(temperature > 0),
                temperature=max(0.1, temperature),  # Avoid division by zero
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids),  # Explicit attention mask
                use_cache=True,  # Enable KV caching for better performance
                repetition_penalty=1.1,  # Slight penalty to avoid repetition
            )

            # Extract new tokens
            new_tokens = outputs[0][input_ids.shape[1] :]
            generated_tokens.append(new_tokens)

            # Update for next iteration
            input_ids = outputs
            remaining -= current_chunk

            # Break if EOS token is generated
            if outputs[0][-1].item() == self.tokenizer.eos_token_id:
                break

        # Combine results and decode
        if generated_tokens:
            combined_tokens = torch.cat(generated_tokens)
            return self.tokenizer.decode(combined_tokens, skip_special_tokens=True)
        else:
            return ""

    def embed_text(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Get text embeddings using the model

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            NumPy array of embeddings
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Use the last hidden state
            last_hidden_state = outputs.last_hidden_state

            # Use mean pooling to get sentence embeddings
            batch_embeddings = []
            for j, input_ids in enumerate(inputs.input_ids):
                # Create attention mask for mean pooling (1 for tokens, 0 for padding)
                attention_mask = inputs.attention_mask[j]

                # Get token embeddings for this text
                token_embeddings = last_hidden_state[j]

                # Apply mean pooling with attention mask
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
                sum_mask = torch.sum(input_mask_expanded, 0)

                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)

                # Calculate mean
                pooled_embedding = sum_embeddings / sum_mask
                batch_embeddings.append(pooled_embedding.cpu().numpy())

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def unload(self):
        """Unload model from memory"""
        del self.model
        self.model = None

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload(self):
        """Reload the model if it was unloaded"""
        if self.model is None:
            self._load_model()


def optimize_transformer_model(
    model_path: str,
    output_path: str,
    model_type: str = "causal_lm",
    sequence_lengths: List[int] = None,
) -> Dict:
    """
    Optimize a transformer model for CPU inference on E5-2640 processors.

    Args:
        model_path: Path to the model directory
        output_path: Path to save optimized model
        model_type: Type of model to optimize ("causal_lm" or "seq2seq")
        sequence_lengths: List of sequence lengths to benchmark

    Returns:
        Dictionary with optimization results
    """
    # Use the TransformerOptimizer to optimize the model
    from core.neural.transformer_optimizer import TransformerOptimizer

    # Initialize optimizer
    optimizer = TransformerOptimizer(model_path, output_path)

    # Run benchmarks and optimization
    if sequence_lengths is None:
        sequence_lengths = [128, 512, 1024]

    results = optimizer.optimize_all(model_type, sequence_lengths)

    # Apply best optimization based on E5-2640 recommendations
    optimized_path = optimizer.apply_best_optimization(model_type)

    # Add the path to results
    results["optimized_model_path"] = optimized_path

    return results
