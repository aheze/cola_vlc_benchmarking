# Google CLI Agent

This agent uses Google's Gemini CLI to solve computer vision tasks through natural language prompts and code generation.

## Overview

The Google CLI Agent leverages the Gemini CLI tool to:
- Understand computer vision task requirements
- Generate Python code solutions using OpenCV
- Execute and refine solutions iteratively
- Produce required output files (e.g., solution.json)

## Prerequisites

1. **Gemini CLI Installation**: The Gemini CLI must be installed and configured:
   ```bash
   npm install -g @google/gemini-cli
   ```

2. **Python Dependencies**: Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Access**: Ensure you have access to Google's Gemini API and the CLI is properly authenticated.

## Usage

### Direct Usage

Run the agent directly on a specific task:

```bash
python run.py --task face_detection_haar --workspace /path/to/workspace
```

### Command Line Options

- `--task`: Task name (e.g., `face_detection_haar`, `coin_detection`)
- `--workspace`: Directory where the agent will work and save results
- `--max_iterations`: Maximum number of refinement iterations (default: 5)
- `--timeout`: Timeout in seconds for CLI commands (default: 300)
- `--verbose`: Enable detailed logging

### Example

```bash
# Run face detection task with verbose output
python run.py --task face_detection_haar --workspace ./test_workspace --verbose

# Run with custom parameters
python run.py --task coin_detection --workspace ./workspace --max_iterations 3 --timeout 600
```

## How It Works

1. **Task Loading**: Loads the task prompt from the corresponding task directory
2. **Prompt Enhancement**: Enhances the prompt with specific instructions for code generation
3. **Gemini Interaction**: Sends the prompt to Gemini CLI and receives a response
4. **Code Extraction**: Extracts Python code blocks from the response
5. **Code Execution**: Executes the generated code in the workspace directory
6. **Iterative Refinement**: If the solution is incomplete, refines the prompt and tries again
7. **Result Validation**: Checks for the required output files (e.g., solution.json)

## Agent Configuration

The agent supports the following hyperparameters:

- `max_iterations`: Maximum number of attempts to solve the task (default: 5)
- `temperature`: Controls randomness in Gemini responses (default: 0.1)
- `timeout`: Timeout for CLI commands in seconds (default: 300)

## Output

The agent produces:
- **solution.json**: Task-specific results in the required format
- **generated_code_*.py**: Generated Python code files
- **Execution logs**: Detailed logs of the solving process (when verbose mode is enabled)

## Integration with Benchmark System

This agent is designed to work with the cola_vlc_benchmarking system:

```bash
# Future integration (when main script is implemented)
python -m benchmark.scripts.run_agent --agent_name google_cli --task face_detection_haar --agent_workspace_dir ./workspace
```

## Troubleshooting

### Common Issues

1. **Gemini CLI not found**:
   - Ensure Gemini CLI is installed: `npm install -g @google/gemini-cli`
   - Check if `gemini` command is available in PATH

2. **Authentication errors**:
   - Verify Gemini API access and authentication
   - Check API key configuration

3. **Code execution failures**:
   - Ensure required Python packages are installed
   - Check workspace directory permissions
   - Verify input files are accessible

4. **Timeout errors**:
   - Increase timeout value for complex tasks
   - Check network connectivity

### Debug Mode

Use the `--verbose` flag to get detailed execution logs:

```bash
python run.py --task face_detection_haar --workspace ./workspace --verbose
```

This will show:
- Task prompt details
- Agent configuration
- Gemini responses
- Code execution results
- Execution token logs

## Supported Tasks

The agent is designed to work with computer vision tasks that require:
- Image processing using OpenCV
- JSON output generation
- File-based input/output operations

Currently tested with:
- Face Detection with Haar Cascades
- Coin Detection
- Homography Estimation

## Limitations

- Requires active internet connection for Gemini API
- Limited by Gemini CLI rate limits and quotas
- Code generation quality depends on prompt clarity
- May require multiple iterations for complex tasks
