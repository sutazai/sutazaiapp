#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SutazAI - Enterprise-grade model management system for Dell PowerEdge R720

This main entry point demonstrates the fully automated model management system,
providing a CLI interface for model downloading, optimization, monitoring, and inference.
"""

import sys
import logging
import argparse
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sutazai.log")],
)

logger = logging.getLogger("sutazai")

# Import SutazAI components
try:
    from core.neural.model_controller import ModelController, generate_text

    MODEL_CONTROLLER_AVAILABLE = True
except ImportError:
    MODEL_CONTROLLER_AVAILABLE = False
    logger.error("Model controller not available - core functionality disabled")


def get_version_info() -> Dict[str, str]:
    """Get SutazAI version information"""
    return {
        "version": "1.0.0",
        "build_date": "2024-10-01",
        "target_hardware": "Dell PowerEdge R720",
        "cpu_target": "Intel E5-2640",
    }


def print_banner():
    """Print SutazAI banner"""
    version_info = get_version_info()
    print("\n" + "=" * 70)
    print(f"SutazAI v{version_info['version']} - Enterprise Model Management System")
    print(
        f"Optimized for {version_info['target_hardware']} with {version_info['cpu_target']} CPUs"
    )
    print("=" * 70 + "\n")


def handle_generate_command(args):
    """Handle the generate command"""
    if not MODEL_CONTROLLER_AVAILABLE:
        print("Error: Model controller not available")
        return 1

    # Initialize controller
    controller = ModelController()

    # Log settings
    logger.info(f"Generating text with model: {args.model or 'auto-selected'}")
    logger.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")

    try:
        # If prompt is from file
        if args.prompt_file:
            try:
                with open(args.prompt_file, "r") as f:
                    prompt = f.read().strip()
            except Exception as e:
                print(f"Error reading prompt file: {e}")
                return 1
        else:
            prompt = args.prompt

        # If system prompt is from file
        system_prompt = None
        if args.system_prompt_file:
            try:
                with open(args.system_prompt_file, "r") as f:
                    system_prompt = f.read().strip()
            except Exception as e:
                print(f"Error reading system prompt file: {e}")
                return 1
        elif args.system_prompt:
            system_prompt = args.system_prompt

        # Generate text
        print("Generating response...", end="", flush=True)

        result = generate_text(
            prompt=prompt,
            model_id=args.model,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print("\r" + " " * 30 + "\r", end="", flush=True)  # Clear the loading message

        if not result.get("success", False):
            print(f"Error: {result.get('error', 'Unknown error')}")
            return 1

        # Print response
        if args.verbose:
            print("\n" + "=" * 50)
            print("RESPONSE:")
            print("=" * 50)

        print(result["response"])

        if args.verbose:
            print("\n" + "=" * 50)
            print(f"Model: {result['model_id']}")
            print(f"Generation time: {result['generation_time_sec']:.2f} seconds")
            if "completion_tokens" in result:
                print(
                    f"Tokens: {result['completion_tokens']} (completion) + {result['prompt_tokens']} (prompt)"
                )
            print("=" * 50)

        # Save to file if requested
        if args.output:
            try:
                with open(args.output, "w") as f:
                    f.write(result["response"])
                print(f"Response saved to: {args.output}")
            except Exception as e:
                print(f"Error saving response: {e}")
                return 1

        return 0

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError during generation: {e}")
        logger.exception("Exception during text generation")
        return 1
    finally:
        # Clean up resources
        controller.cleanup()


def handle_models_command(args):
    """Handle the models command"""
    if not MODEL_CONTROLLER_AVAILABLE:
        print("Error: Model controller not available")
        return 1

    # Initialize controller
    controller = ModelController()

    try:
        if args.download:
            # Download a model
            print(f"Downloading model: {args.download}")

            model_result = controller.get_model(
                model_id=args.download, force_reload=args.force
            )

            if model_result.get("success", False):
                model_info = model_result.get("model_info", {})
                print("Model downloaded successfully!")
                print(f"Path: {model_info.get('model_path')}")

                if model_info.get("optimized_path"):
                    print("Optimized: Yes")
                    print(f"Optimized path: {model_info.get('optimized_path')}")
                else:
                    print("Optimized: No")

                if "metrics" in model_info:
                    print(
                        f"Performance: {model_info['metrics'].get('tokens_per_second', 'N/A')} tokens/sec"
                    )
            else:
                print(f"Error: {model_result.get('error', 'Unknown error')}")
                return 1

        elif args.recommended:
            # Get recommended model
            recommended = controller.get_recommended_model()

            if recommended.get("success", False):
                print(f"Recommended model for this system: {recommended['model_id']}")
                print(
                    f"Status: {'Loaded' if recommended['is_loaded'] else 'Not loaded'}"
                )
                print(
                    f"\nRun with --download {recommended['model_id']} to download this model"
                )
            else:
                print(f"Error: {recommended.get('error', 'Unknown error')}")
                return 1

        elif args.info:
            # Get model info
            result = controller.get_model(args.info)

            if not result.get("success", False):
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1

            model_info = result.get("model_info", {})
            print(f"Model: {args.info}")
            print(f"Path: {model_info.get('model_path')}")

            if model_info.get("optimized_path"):
                print("Optimized: Yes")
                print(f"Optimized path: {model_info.get('optimized_path')}")
            else:
                print("Optimized: No")

            if "metrics" in model_info:
                metrics = model_info["metrics"]
                print("\nPerformance metrics:")
                print(f"  Inference time: {metrics.get('inference_time_ms', 'N/A')} ms")
                print(f"  Speed: {metrics.get('tokens_per_second', 'N/A')} tokens/sec")
                print(f"  Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")

        else:
            # List models
            result = controller.list_models()

            if result.get("success", False):
                models = result.get("models", [])
                print(f"Available models ({len(models)}):")

                for model in models:
                    status = "✓" if model["available"] else "✗"
                    loaded = "[loaded]" if model["loaded"] else ""
                    opt = "[optimized]" if model["optimized"] else ""
                    print(f"  {status} {model['id']} {loaded} {opt}")

                    if args.verbose and "performance" in model:
                        perf = model["performance"]
                        if perf.get("tokens_per_second"):
                            print(
                                f"    Speed: {perf['tokens_per_second']:.1f} tokens/sec"
                            )

                loaded_models = result.get("loaded_models", [])
                print(
                    f"\nLoaded models: {', '.join(loaded_models) if loaded_models else 'None'}"
                )
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception during models command")
        return 1
    finally:
        # Clean up resources
        controller.cleanup()


def main():
    """Main entry point"""
    # Print banner
    print_banner()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="SutazAI - Enterprise Model Management System",
        epilog="Optimized for Dell PowerEdge R720 with E5-2640 CPUs",
    )

    # Global options
    parser.add_argument(
        "--version", action="store_true", help="Show version information"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text with a model"
    )
    generate_parser.add_argument("prompt", nargs="?", help="Text prompt for generation")
    generate_parser.add_argument("--prompt-file", "-f", help="Read prompt from file")
    generate_parser.add_argument("--system-prompt", "-s", help="System prompt")
    generate_parser.add_argument(
        "--system-prompt-file", help="Read system prompt from file"
    )
    generate_parser.add_argument("--model", "-m", help="Model ID to use")
    generate_parser.add_argument(
        "--max-tokens", "-t", type=int, default=512, help="Maximum tokens to generate"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    generate_parser.add_argument("--output", "-o", help="Save response to file")

    # Models command
    models_parser = subparsers.add_parser("models", help="Manage models")
    models_parser.add_argument("--download", "-d", help="Download a specific model")
    models_parser.add_argument("--info", "-i", help="Show detailed info for a model")
    models_parser.add_argument(
        "--recommended", "-r", action="store_true", help="Show recommended model"
    )
    models_parser.add_argument(
        "--force", action="store_true", help="Force redownload/reload"
    )

    # Parse arguments
    args = parser.parse_args()

    # Show version if requested
    if args.version:
        version_info = get_version_info()
        print(f"SutazAI v{version_info['version']} ({version_info['build_date']})")
        print(
            f"Target: {version_info['target_hardware']} with {version_info['cpu_target']} CPUs"
        )
        return 0

    # Handle commands
    if args.command == "generate":
        if not args.prompt and not args.prompt_file:
            generate_parser.error("Either prompt or --prompt-file is required")
        return handle_generate_command(args)

    elif args.command == "models":
        return handle_models_command(args)

    else:
        # No command provided, show help
        parser.print_help()
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unhandled exception")
        print(f"Error: {e}")
        sys.exit(1)
