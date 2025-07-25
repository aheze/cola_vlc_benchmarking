import os
import sys
import json
import subprocess
import tempfile
from typing import Any, Dict, Optional, List
from pathlib import Path

# Add the project root to the path to import BaseAgent
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from benchmark.core.base_agent import BaseAgent


class GoogleCLIAgent(BaseAgent):
    """
    Google CLI (Gemini) Agent that uses the Gemini CLI to solve computer vision tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Google CLI agent.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.agent_name = "Google CLI (Gemini)"
        self.agent_description = "Agent that uses Google's Gemini CLI to solve computer vision tasks"
        
        # Default hyperparameters
        self.hyperparameters = {
            "max_iterations": 5,
            "temperature": 0.1,
            "timeout": 300,  # 5 minutes timeout for CLI commands
        }
        
        # Update with any provided hyperparameters
        if config and "hyperparameters" in config:
            self.hyperparameters.update(config["hyperparameters"])
    
    def run_task(self, prompt: str, path: str, tools: Optional[List[str]] = None, task_name: Optional[str] = None) -> Any:
        """
        Run the agent with the given prompt in the specified directory.
        
        Args:
            prompt: The task prompt/instruction
            path: The working directory path for the task
            tools: Optional list of specific tools to use for this task
            task_name: Optional task name to locate input files
            
        Returns:
            Dict: The result of running the task
        """
        self.ensure_setup()
        self.clear_run_tokens()
        
        # Log the start of the task
        self.log_token("task_start", {
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "workspace_path": path,
            "tools": tools
        })
        
        try:
            # Ensure workspace directory exists
            os.makedirs(path, exist_ok=True)
            
            # Change to the workspace directory
            original_cwd = os.getcwd()
            os.chdir(path)
            
            try:
                # Run the task using Gemini CLI
                result = self._run_gemini_task(prompt, path, task_name)
                
                self.log_token("task_complete", {
                    "success": True,
                    "result": result
                })
                
                return result
                
            finally:
                # Always restore the original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            error_msg = f"Error running task: {str(e)}"
            self.log_token("task_error", {
                "error": error_msg,
                "success": False
            })
            return {
                "success": False,
                "error": error_msg,
                "output": None
            }
    
    def _run_gemini_task(self, prompt: str, workspace_path: str, task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the task using Gemini CLI with iterative refinement.
        
        Args:
            prompt: The task prompt
            workspace_path: Path to the workspace directory
            task_name: Optional task name to locate input files
            
        Returns:
            Dict containing the task result
        """
        # Enhanced prompt with specific instructions for code generation
        enhanced_prompt = self._create_enhanced_prompt(prompt, workspace_path, task_name)
        
        max_iterations = self.hyperparameters.get("max_iterations", 5)
        
        for iteration in range(max_iterations):
            self.log_token("iteration_start", {
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })
            
            try:
                # Run Gemini CLI
                response = self._call_gemini_cli(enhanced_prompt)
                
                self.log_token("gemini_response", {
                    "iteration": iteration + 1,
                    "response_length": len(response),
                    "response_preview": response[:500] + "..." if len(response) > 500 else response
                })
                
                # Try to extract and execute code from the response
                success = self._process_gemini_response(response, workspace_path)
                
                if success:
                    # Check if solution.json was created
                    solution_path = os.path.join(workspace_path, "solution.json")
                    if os.path.exists(solution_path):
                        self.log_token("solution_found", {
                            "iteration": iteration + 1,
                            "solution_path": solution_path
                        })
                        
                        return {
                            "success": True,
                            "iterations": iteration + 1,
                            "output": "Solution generated successfully",
                            "solution_path": solution_path
                        }
                
                # If not successful, prepare for next iteration
                if iteration < max_iterations - 1:
                    enhanced_prompt = self._create_refinement_prompt(prompt, response, workspace_path)
                
            except Exception as e:
                self.log_token("iteration_error", {
                    "iteration": iteration + 1,
                    "error": str(e)
                })
                
                if iteration == max_iterations - 1:
                    raise e
                
                # Continue to next iteration
                continue
        
        # If we get here, all iterations failed
        return {
            "success": False,
            "iterations": max_iterations,
            "output": "Failed to generate solution after maximum iterations",
            "error": "Maximum iterations reached without success"
        }
    
    def _create_enhanced_prompt(self, original_prompt: str, workspace_path: str, task_name: Optional[str] = None) -> str:
        """
        Create an enhanced prompt with additional context and instructions.
        
        Args:
            original_prompt: The original task prompt
            workspace_path: Path to the workspace directory
            task_name: Optional task name to locate input files
            
        Returns:
            Enhanced prompt string
        """
        # Check for input files
        input_files = []
        input_dir = None
        
        # If task_name is provided, look for input files in the task directory
        if task_name:
            # Find the project root and look for task input directory
            current_dir = Path(workspace_path)
            project_root = None
            
            # Look for benchmark directory in parent directories
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "benchmark").exists():
                    project_root = parent
                    break
            
            if project_root:
                task_input_dir = project_root / "benchmark" / "tasks" / task_name / "input"
                if task_input_dir.exists():
                    input_dir = task_input_dir
                    input_files = [f.name for f in task_input_dir.iterdir() if f.is_file()]
        
        # Fallback: Look for input directory in parent directories (original behavior)
        if not input_dir:
            current_dir = Path(workspace_path)
            for parent in [current_dir] + list(current_dir.parents):
                potential_input_dir = parent / "input"
                if potential_input_dir.exists():
                    input_dir = potential_input_dir
                    input_files = [f.name for f in potential_input_dir.iterdir() if f.is_file()]
                    break
        
        enhanced_prompt = f"""
You are a computer vision expert. Please solve the following task:

{original_prompt}

IMPORTANT INSTRUCTIONS:
1. You must write Python code to solve this task
2. The code should be complete and executable
3. Use OpenCV (cv2) for computer vision operations
4. Save the final results in a file named 'solution.json' in the current directory
5. Make sure to handle all edge cases and error conditions
6. The code should be self-contained and not require user input
7. CRITICAL: Use the exact input directory path provided below - do NOT assume it's in the current directory

WORKSPACE INFORMATION:
- Current working directory: {workspace_path}
- Input files available: {input_files if input_files else 'None found'}
- Input directory (USE THIS EXACT PATH): {input_dir if input_dir else 'Not found'}

IMPORTANT: The input images are located at the path: {input_dir}
Make sure your code uses this exact path to read the images, not a relative 'input' directory.

Please provide the complete Python code to solve this task. Start your response with the Python code wrapped in ```python code blocks.
"""
        
        return enhanced_prompt
    
    def _create_refinement_prompt(self, original_prompt: str, previous_response: str, workspace_path: str) -> str:
        """
        Create a refinement prompt based on previous attempt.
        
        Args:
            original_prompt: The original task prompt
            previous_response: Previous Gemini response
            workspace_path: Path to the workspace directory
            
        Returns:
            Refinement prompt string
        """
        # Check what files exist in workspace
        existing_files = []
        if os.path.exists(workspace_path):
            existing_files = [f for f in os.listdir(workspace_path) if os.path.isfile(os.path.join(workspace_path, f))]
        
        refinement_prompt = f"""
The previous attempt did not fully complete the task. Please refine the solution.

ORIGINAL TASK:
{original_prompt}

PREVIOUS ATTEMPT RESPONSE:
{previous_response[:1000]}...

CURRENT WORKSPACE STATUS:
- Files created: {existing_files}
- Missing: solution.json (if not in list above)

Please provide a corrected and complete Python solution. Focus on:
1. Ensuring all required output files are created
2. Fixing any errors from the previous attempt
3. Making sure the solution.json format matches the requirements exactly

Provide the complete Python code wrapped in ```python code blocks.
"""
        
        return refinement_prompt
    
    def _call_gemini_cli(self, prompt: str) -> str:
        """
        Call the Gemini CLI with the given prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            The response from Gemini CLI
        """
        timeout = self.hyperparameters.get("timeout", 300)
        
        try:
            # Create a temporary file for the prompt to handle long prompts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            try:
                print(f"\n🤖 Calling Gemini CLI...")
                print(f"📝 Prompt length: {len(prompt)} characters")
                print(f"📝 Prompt: {prompt}")
                print("=" * 60)
                
                # Use Popen to stream output in real-time
                # Read the prompt from file and pass it via -p flag
                with open(prompt_file, 'r') as f:
                    prompt_content = f.read()
                
                process = subprocess.Popen(
                    ['gemini', '-p', prompt_content],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Collect output while streaming to terminal
                output_lines = []
                try:
                    if process.stdout:
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            print(line.rstrip())  # Print to terminal in real-time
                            output_lines.append(line)
                    
                    # Wait for process to complete
                    return_code = process.wait(timeout=timeout)
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    raise RuntimeError(f"Gemini CLI timed out after {timeout} seconds")
                
                output = ''.join(output_lines)
                
                print("=" * 60)
                print(f"✅ Gemini CLI completed with return code: {return_code}")
                
                if return_code != 0:
                    print(f"❌ Error output: {output}")
                    raise RuntimeError(f"Gemini CLI failed with return code {return_code}: {output}")
                
                return output.strip()
                
            finally:
                # Clean up the temporary prompt file
                try:
                    os.unlink(prompt_file)
                except:
                    pass
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Gemini CLI timed out after {timeout} seconds")
        except FileNotFoundError:
            raise RuntimeError("Gemini CLI not found. Please ensure it's installed with: npm install -g @google/gemini-cli")
    
    def _process_gemini_response(self, response: str, workspace_path: str) -> bool:
        """
        Process the Gemini response and execute any Python code found.
        
        Args:
            response: The response from Gemini
            workspace_path: Path to the workspace directory
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Extract Python code blocks from the response
            code_blocks = self._extract_code_blocks(response)
            
            if not code_blocks:
                self.log_token("no_code_found", {"response_preview": response[:200]})
                return False
            
            # Execute each code block
            for i, code in enumerate(code_blocks):
                self.log_token("executing_code", {
                    "block_index": i,
                    "code_preview": code[:200] + "..." if len(code) > 200 else code
                })
                
                # Save code to a temporary file and execute it
                code_file = os.path.join(workspace_path, f"generated_code_{i}.py")
                with open(code_file, 'w') as f:
                    f.write(code)
                
                # Execute the code
                result = subprocess.run(
                    [sys.executable, code_file],
                    capture_output=True,
                    text=True,
                    cwd=workspace_path,
                    timeout=self.hyperparameters.get("timeout", 300)
                )
                
                if result.returncode != 0:
                    self.log_token("code_execution_error", {
                        "block_index": i,
                        "error": result.stderr,
                        "stdout": result.stdout
                    })
                    # Don't return False immediately, try other code blocks
                    continue
                else:
                    self.log_token("code_execution_success", {
                        "block_index": i,
                        "stdout": result.stdout
                    })
            
            return True
            
        except Exception as e:
            self.log_token("processing_error", {"error": str(e)})
            return False
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract Python code blocks from markdown-formatted text.
        
        Args:
            text: The text containing code blocks
            
        Returns:
            List of extracted code strings
        """
        import re
        
        # Pattern to match ```python ... ``` blocks
        pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        # Clean up the code blocks
        code_blocks = []
        for match in matches:
            # Remove leading/trailing whitespace and ensure it looks like Python code
            code = match.strip()
            if code and ('import' in code or 'def' in code or 'cv2' in code or 'json' in code):
                code_blocks.append(code)
        
        return code_blocks
    
    def get_info(self) -> Dict[str, Any]:
        """
        Returns information about the agent.
        
        Returns:
            Dict containing agent information
        """
        return {
            "name": self.agent_name,
            "description": self.agent_description,
            "version": "1.0.0",
            "capabilities": [
                "Computer Vision Tasks",
                "Code Generation",
                "Iterative Problem Solving",
                "OpenCV Integration"
            ],
            "hyperparameters": self.hyperparameters,
            "tools_enabled": self.get_enabled_tools(),
            "setup_status": self.is_setup
        }
    
    def set_hyperparameters(self, **kwargs) -> None:
        """
        Set or update hyperparameters for the agent.
        
        Args:
            **kwargs: Hyperparameter key-value pairs
        """
        valid_params = ["max_iterations", "temperature", "timeout"]
        
        for key, value in kwargs.items():
            if key in valid_params:
                self.hyperparameters[key] = value
                self.log_token("hyperparameter_update", {
                    "parameter": key,
                    "value": value
                })
            else:
                print(f"Warning: Unknown hyperparameter '{key}' ignored")
        
        print(f"Updated hyperparameters: {self.hyperparameters}")
