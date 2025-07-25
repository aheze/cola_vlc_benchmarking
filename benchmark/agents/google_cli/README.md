# Google CLI Agent

A computer vision agent that uses Google's Gemini CLI to solve tasks through natural language prompts and code generation.

## Quick Start

1. **Install Gemini CLI**:
   ```bash
   ./setup.sh
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a task**:
   ```bash
   python3 run.py --task face_detection_haar --workspace ./workspace --verbose
   ```

## Usage

```bash
python3run.py --task TASK_NAME --workspace WORKSPACE_DIR [OPTIONS]
```

### Options
- `--task`: Task to run (e.g., `face_detection_haar`, `coin_detection`)
- `--workspace`: Directory for agent work and results
- `--max_iterations`: Max attempts (default: 5)
- `--timeout`: Command timeout in seconds (default: 300)
- `--verbose`: Show detailed logs

### Example
```bash
# Run face detection with detailed output
python3 run.py --task face_detection_haar --workspace ./test --verbose

# Run with custom settings
python3 run.py --task coin_detection --workspace ./work --max_iterations 3
```

## How It Works

1. Loads task prompt from `benchmark/tasks/{task_name}/prompt.md`
2. Enhances prompt with workspace info and input file locations
3. Sends prompt to Gemini CLI
4. Extracts and executes generated Python code
5. Iteratively refines solution if needed
6. Produces `solution.json` with results

## Output Files

- `solution.json` - Task results in required format
- `generated_code_*.py` - Generated Python scripts (* is the ith generated code block)
- Execution logs (with `--verbose`)

## Supported Tasks

✅ **face_detection_haar** - Detect faces using Haar cascades  
✅ **coin_detection** - Detect and count coins  
✅ **homography_estimation** - Estimate image homography  
