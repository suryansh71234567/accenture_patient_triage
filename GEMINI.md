# Workspace Execution Rules

This workspace enforces that all Python commands, package installations, and script executions run strictly within the local virtual environment.

## Rules
1. **Virtual Environment Requirement**:
   - You MUST run all Python scripts and commands using the virtual environment located at `.venv` in the workspace root.
   - Do NOT use global python or global pip commands.
2. **Command Reference**:
   - **Running Python**: Use `.venv\Scripts\python.exe <args>` or activate the environment first.
   - **Installing packages**: Use `.venv\Scripts\pip.exe install <packages>` or `.venv\Scripts\python.exe -m pip install <packages>`.
3. **PowerShell Execution**:
   - Before executing code or running interactive Python commands, ensure the environment is activated or reference the exact executables in the `.venv\Scripts` directory.
