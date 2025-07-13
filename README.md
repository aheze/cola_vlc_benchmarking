To run a task manually, 
1. Create a workspace folder for the agent. 
2. Take the required files that the tasks requires and copy over into the task workspace folder
3. Use the task's prompt
4. To evaluation run: `python -m benchmark.runners.task_runner --task <task_name> --solution_path <task workspace folder path>`

Available Tasks:
- coin_detection

