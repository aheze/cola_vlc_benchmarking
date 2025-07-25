#!/usr/bin/env python3
"""
Run script for Google CLI Agent.

This script allows running the Google CLI agent on a specific task.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.agents.google_cli.google_cli_agent import GoogleCLIAgent


def load_task_prompt(task_name: str) -> str:
    """
    Load the prompt for a given task.
    
    Args:
        task_name: Name of the task (e.g., 'face_detection_haar')
        
    Returns:
        The task prompt as a string
    """
    # Find the task directory
    script_dir = Path(__file__).parent
    task_dir = script_dir.parent.parent / "tasks" / task_name
    prompt_file = task_dir / "prompt.md"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Task prompt not found: {prompt_file}")
    
    with open(prompt_file, 'r') as f:
        return f.read()


def main():
    """Main function to run the Google CLI agent."""
    parser = argparse.ArgumentParser(description="Run Google CLI Agent on a specific task")
    parser.add_argument("--task", type=str, required=True, 
                       help="Task name (e.g., face_detection_haar)")
    parser.add_argument("--workspace", type=str, required=True,
                       help="Workspace directory path for the agent")
    parser.add_argument("--max_iterations", type=int, default=5,
                       help="Maximum number of iterations (default: 5)")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout in seconds for CLI commands (default: 300)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Load the task prompt
        print(f"Loading task: {args.task}")
        prompt = load_task_prompt(args.task)
        
        if args.verbose:
            print(f"Task prompt preview: {prompt[:200]}...")
        
        # Create agent configuration
        config = {
            "name": "Google CLI Agent",
            "description": "Agent using Google Gemini CLI for computer vision tasks",
            "hyperparameters": {
                "max_iterations": args.max_iterations,
                "timeout": args.timeout
            }
        }
        
        # Initialize the agent
        print("Initializing Google CLI Agent...")
        agent = GoogleCLIAgent(config)
        agent.setup()
        
        if args.verbose:
            print(f"Agent info: {agent.get_info()}")
        
        # Create workspace directory
        workspace_path = os.path.abspath(args.workspace)
        os.makedirs(workspace_path, exist_ok=True)
        print(f"Workspace: {workspace_path}")
        
        # Run the task
        print("Running task...")
        result = agent.run_task(prompt, workspace_path)
        
        # Print results
        print("\n" + "="*50)
        print("TASK EXECUTION RESULTS")
        print("="*50)
        
        if result.get("success", False):
            print("✅ Task completed successfully!")
            print(f"Iterations used: {result.get('iterations', 'N/A')}")
            print(f"Output: {result.get('output', 'N/A')}")
            
            solution_path = result.get('solution_path')
            if solution_path and os.path.exists(solution_path):
                print(f"Solution file created: {solution_path}")
                
                # Show solution preview if verbose
                if args.verbose:
                    try:
                        with open(solution_path, 'r') as f:
                            solution_content = f.read()
                        print(f"Solution preview: {solution_content[:500]}...")
                    except Exception as e:
                        print(f"Could not read solution file: {e}")
        else:
            print("❌ Task failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Output: {result.get('output', 'N/A')}")
        
        # Show execution tokens if verbose
        if args.verbose:
            print("\nExecution Log:")
            tokens = agent.get_run_tokens()
            for i, token in enumerate(tokens[-10:]):  # Show last 10 tokens
                print(f"  {i+1}. [{token['token_type']}] {token.get('content', {})}")
        
        print("="*50)
        
        # Return appropriate exit code
        return 0 if result.get("success", False) else 1
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
