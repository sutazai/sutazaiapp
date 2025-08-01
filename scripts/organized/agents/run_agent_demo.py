#!/usr/bin/env python3
"""
Quick Runner for SutazAI Agent Demo
==================================

This is a simplified runner that handles system checks and provides
different demo modes for various use cases.

Usage:
    python run_agent_demo.py [mode]
    
Modes:
    basic       - Basic agent creation and communication
    full        - Complete demo with all features (default)
    custom      - Interactive custom workflow
    benchmark   - Performance benchmarking
    monitoring  - System monitoring demo
"""

import asyncio
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required services are running"""
    issues = []
    
    # Check Redis
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        logger.info("✅ Redis server is running")
    except Exception as e:
        issues.append(f"❌ Redis server not available: {e}")
    
    # Check Ollama
    try:
        import httpx
        with httpx.Client() as client:
            response = client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                logger.info(f"✅ Ollama server running with models: {models}")
                
                # Check for required models
                required_models = ["codellama", "llama2"]
                missing_models = [m for m in required_models if not any(m in model for model in models)]
                
                if missing_models:
                    issues.append(f"⚠️  Missing Ollama models: {missing_models}")
                    logger.info("Run: ollama pull codellama && ollama pull llama2")
            else:
                issues.append("❌ Ollama server not responding properly")
    except Exception as e:
        issues.append(f"❌ Ollama server not available: {e}")
    
    # Check Python dependencies
    try:
        import aioredis
        import httpx
        logger.info("✅ Python dependencies available")
    except ImportError as e:
        issues.append(f"❌ Missing Python dependency: {e}")
        logger.info("Run: pip install -r agent_demo_requirements.txt")
    
    return issues


async def run_basic_demo():
    """Run basic agent demo"""
    logger.info("🚀 Running Basic Agent Demo")
    
    from sutazai_agent_demo import SutazAIAgentDemo
    
    demo = SutazAIAgentDemo()
    
    try:
        # Initialize system
        if not await demo.initialize():
            logger.error("Failed to initialize demo system")
            return False
        
        # Create a few agents
        agents = await demo.create_all_agents()
        logger.info(f"Created {len(agents)} agents")
        
        # Basic communication test
        if len(agents) >= 2:
            await demo.demonstrate_agent_communication()
            logger.info("✅ Agent communication test passed")
        
        # Basic task execution
        await demo.demonstrate_task_execution()
        logger.info("✅ Task execution test passed")
        
        logger.info("🎉 Basic demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Basic demo failed: {e}")
        return False
    finally:
        await demo.cleanup()


async def run_full_demo():
    """Run complete demo with all features"""
    logger.info("🚀 Running Full Agent Demo")
    
    from sutazai_agent_demo import SutazAIAgentDemo
    
    demo = SutazAIAgentDemo()
    
    try:
        report = await demo.run_complete_demo()
        logger.info("🎉 Full demo completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("DEMO SUMMARY")
        print("="*60)
        print(f"Session ID: {demo.demo_session_id}")
        print(f"Agents Created: {len(demo.active_agents)}")
        print(f"Tasks Executed: {len(demo.task_results)}")
        
        successful_tasks = len([
            r for r in demo.task_results.values() 
            if isinstance(r, dict) and r.get("success")
        ])
        print(f"Successful Tasks: {successful_tasks}")
        
        if demo.performance_metrics:
            print(f"System Health: {'✅ Good' if demo.performance_metrics.get('successful_tasks', 0) > 0 else '⚠️  Issues'}")
        
        print("="*60)
        return True
        
    except Exception as e:
        logger.error(f"Full demo failed: {e}")
        return False


async def run_custom_demo():
    """Run interactive custom demo"""
    logger.info("🚀 Running Custom Interactive Demo")
    
    from sutazai_agent_demo import SutazAIAgentDemo
    
    demo = SutazAIAgentDemo()
    
    try:
        # Initialize system
        if not await demo.initialize():
            logger.error("Failed to initialize demo system")
            return False
        
        # Create agents
        agents = await demo.create_all_agents()
        if not agents:
            logger.error("No agents created")
            return False
        
        print(f"\n✅ Created {len(agents)} agents")
        print("\nAvailable agents:")
        for i, (agent_id, agent) in enumerate(agents.items(), 1):
            print(f"  {i}. {agent_id} - {agent.name}")
        
        # Interactive loop
        while True:
            print("\n" + "="*50)
            print("CUSTOM DEMO OPTIONS")
            print("="*50)
            print("1. Send message between agents")
            print("2. Execute task on agent")
            print("3. Check agent status")
            print("4. View system metrics")
            print("5. Run collaborative workflow")
            print("0. Exit")
            
            choice = input("\nSelect option (0-5): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await interactive_messaging(demo)
            elif choice == "2":
                await interactive_task_execution(demo)
            elif choice == "3":
                await show_agent_status(demo)
            elif choice == "4":
                await show_system_metrics(demo)
            elif choice == "5":
                await demo.demonstrate_collaborative_workflow()
                logger.info("✅ Collaborative workflow completed")
            else:
                print("Invalid option")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return True
    except Exception as e:
        logger.error(f"Custom demo failed: {e}")
        return False
    finally:
        await demo.cleanup()


async def interactive_messaging(demo):
    """Interactive agent messaging"""
    agents = list(demo.active_agents.items())
    
    print(f"\nSelect sender agent:")
    for i, (agent_id, agent) in enumerate(agents, 1):
        print(f"  {i}. {agent_id}")
    
    try:
        sender_idx = int(input("Sender (number): ")) - 1
        if sender_idx < 0 or sender_idx >= len(agents):
            print("Invalid selection")
            return
        
        sender_id, sender_agent = agents[sender_idx]
        
        print(f"\nSelect receiver agent (or 0 for broadcast):")
        print("  0. Broadcast to all")
        for i, (agent_id, agent) in enumerate(agents, 1):
            if agent_id != sender_id:
                print(f"  {i}. {agent_id}")
        
        receiver_choice = input("Receiver (number): ").strip()
        
        if receiver_choice == "0":
            receiver_id = "broadcast"
        else:
            receiver_idx = int(receiver_choice) - 1
            if receiver_idx < 0 or receiver_idx >= len(agents):
                print("Invalid selection")
                return
            receiver_id = agents[receiver_idx][0]
        
        message = input("Message content: ").strip()
        
        message_id = await sender_agent.send_message(
            receiver_id,
            "custom_message",
            {"content": message, "demo": True}
        )
        
        print(f"✅ Message sent with ID: {message_id}")
        
    except (ValueError, IndexError):
        print("Invalid input")
    except Exception as e:
        logger.error(f"Messaging failed: {e}")


async def interactive_task_execution(demo):
    """Interactive task execution"""
    agents = list(demo.active_agents.items())
    
    print(f"\nSelect agent for task execution:")
    for i, (agent_id, agent) in enumerate(agents, 1):
        capabilities = [cap.value for cap in agent.capabilities]
        print(f"  {i}. {agent_id} - Capabilities: {capabilities}")
    
    try:
        agent_idx = int(input("Agent (number): ")) - 1
        if agent_idx < 0 or agent_idx >= len(agents):
            print("Invalid selection")
            return
        
        agent_id, agent = agents[agent_idx]
        
        print(f"\nTask types for {agent_id}:")
        if "code_generation" in [cap.value for cap in agent.capabilities]:
            print("  1. generate_code")
            print("  2. explain_code")
            print("  3. refactor_code")
        print("  4. custom_task")
        
        task_type = input("Task type: ").strip()
        
        if task_type == "generate_code" or task_type == "1":
            spec = input("Code specification: ").strip()
            language = input("Language (python): ").strip() or "python"
            
            task_data = {
                "task_type": "generate_code",
                "specification": spec,
                "language": language,
                "code_type": "function"
            }
        else:
            task_data = {
                "task_type": "custom_task",
                "description": input("Task description: ").strip()
            }
        
        print("Executing task...")
        result = await agent.execute_task(f"interactive_{len(demo.task_results)}", task_data)
        
        if result.get("success"):
            print("✅ Task completed successfully!")
            if "generated_code" in result.get("result", {}):
                code = result["result"]["generated_code"]
                print(f"\nGenerated code:\n{code[:500]}{'...' if len(code) > 500 else ''}")
        else:
            print(f"❌ Task failed: {result.get('error')}")
        
        demo.task_results[f"interactive_{len(demo.task_results)}"] = result
        
    except (ValueError, IndexError):
        print("Invalid input")
    except Exception as e:
        logger.error(f"Task execution failed: {e}")


async def show_agent_status(demo):
    """Show status of all agents"""
    print(f"\n{'='*60}")
    print("AGENT STATUS")
    print(f"{'='*60}")
    
    for agent_id, agent in demo.active_agents.items():
        try:
            info = await agent.get_agent_info()
            print(f"\n{agent_id}:")
            print(f"  Status: {info['status']}")
            print(f"  Active Tasks: {info['active_tasks']}")
            print(f"  Total Tasks: {info['task_count']}")
            print(f"  Errors: {info['error_count']}")
            print(f"  Uptime: {info['uptime']:.1f}s")
        except Exception as e:
            print(f"\n{agent_id}: Error getting status - {e}")


async def show_system_metrics(demo):
    """Show system metrics"""
    print(f"\n{'='*60}")
    print("SYSTEM METRICS")
    print(f"{'='*60}")
    
    try:
        # Registry stats
        if demo.agent_registry:
            stats = demo.agent_registry.get_registry_stats()
            print(f"\nAgent Registry:")
            print(f"  Total Agents: {stats['total_agents']}")
            print(f"  Status Counts: {stats['status_counts']}")
        
        # Communication bus metrics
        if demo.communication_bus:
            metrics = await demo.communication_bus.get_system_metrics()
            print(f"\nCommunication Bus:")
            print(f"  Task Queue Size: {metrics['task_queue_size']}")
            print(f"  Priority Queue Size: {metrics['priority_queue_size']}")
            print(f"  Active Agents: {metrics['active_agents']}")
            print(f"  Average Load: {metrics['average_agent_load']:.2f}")
        
        # Task results
        successful = len([r for r in demo.task_results.values() 
                         if isinstance(r, dict) and r.get("success")])
        total = len(demo.task_results)
        print(f"\nTask Execution:")
        print(f"  Total Tasks: {total}")
        print(f"  Successful: {successful}")
        print(f"  Success Rate: {(successful/total*100) if total > 0 else 0:.1f}%")
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")


async def run_benchmark():
    """Run performance benchmark"""
    logger.info("🚀 Running Performance Benchmark")
    
    from sutazai_agent_demo import SutazAIAgentDemo
    import time
    
    demo = SutazAIAgentDemo()
    
    try:
        start_time = time.time()
        
        # Initialize
        init_start = time.time()
        await demo.initialize()
        init_time = time.time() - init_start
        
        # Create agents
        agent_start = time.time()
        agents = await demo.create_all_agents()
        agent_time = time.time() - agent_start
        
        # Run tasks concurrently
        task_start = time.time()
        tasks = []
        
        for i in range(10):  # Run 10 concurrent tasks
            if agents:
                agent = list(agents.values())[i % len(agents)]
                task = agent.execute_task(f"benchmark_{i}", {
                    "task_type": "generate_code",
                    "specification": f"Create a simple function #{i}",
                    "language": "python"
                })
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        task_time = time.time() - task_start
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = len([r for r in results if isinstance(r, dict) and r.get("success")])
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Initialization Time: {init_time:.2f}s")
        print(f"Agent Creation Time: {agent_time:.2f}s")
        print(f"Task Execution Time: {task_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Agents Created: {len(agents)}")
        print(f"Tasks Executed: {len(results)}")
        print(f"Successful Tasks: {successful}")
        print(f"Success Rate: {(successful/len(results)*100) if results else 0:.1f}%")
        print(f"Tasks per Second: {len(results)/task_time:.2f}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return False
    finally:
        await demo.cleanup()


async def run_monitoring_demo():
    """Run system monitoring demo"""
    logger.info("🚀 Running System Monitoring Demo")
    
    from sutazai_agent_demo import SutazAIAgentDemo
    
    demo = SutazAIAgentDemo()
    
    try:
        await demo.initialize()
        agents = await demo.create_all_agents()
        
        print(f"Created {len(agents)} agents. Monitoring for 30 seconds...")
        
        for i in range(6):  # Monitor for 6 intervals of 5 seconds each
            await asyncio.sleep(5)
            
            print(f"\n--- Monitoring Cycle {i+1} ---")
            await show_system_metrics(demo)
            
            # Execute some background tasks
            if agents and i < 3:
                agent = list(agents.values())[0]
                asyncio.create_task(agent.execute_task(f"monitor_task_{i}", {
                    "task_type": "generate_code",
                    "specification": f"Background task {i}",
                    "language": "python"
                }))
        
        print("\n✅ Monitoring demo completed")
        return True
        
    except Exception as e:
        logger.error(f"Monitoring demo failed: {e}")
        return False
    finally:
        await demo.cleanup()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SutazAI Agent Demo Runner")
    parser.add_argument(
        "mode", 
        nargs="?", 
        default="full",
        choices=["basic", "full", "custom", "benchmark", "monitoring"],
        help="Demo mode to run"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip dependency checks"
    )
    
    args = parser.parse_args()
    
    print("🤖 SutazAI Agent System Demo")
    print("="*50)
    
    # Check dependencies unless skipped
    if not args.skip_checks:
        logger.info("Checking system dependencies...")
        issues = check_dependencies()
        
        if issues:
            print("\n⚠️  System Issues Found:")
            for issue in issues:
                print(f"  {issue}")
            
            if any("❌" in issue for issue in issues):
                print("\n🛑 Critical issues found. Please fix them before running the demo.")
                return 1
            else:
                print("\n⚠️  Some issues found, but demo may still work.")
    
    # Run selected demo mode
    try:
        success = False
        
        if args.mode == "basic":
            success = await run_basic_demo()
        elif args.mode == "full":
            success = await run_full_demo()
        elif args.mode == "custom":
            success = await run_custom_demo()
        elif args.mode == "benchmark":
            success = await run_benchmark()
        elif args.mode == "monitoring":
            success = await run_monitoring_demo()
        
        if success:
            print(f"\n🎉 {args.mode.title()} demo completed successfully!")
            return 0
        else:
            print(f"\n❌ {args.mode.title()} demo failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Demo failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)