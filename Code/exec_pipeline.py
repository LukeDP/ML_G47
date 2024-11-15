#! /usr/bin/env python3
import subprocess
import os

pipeline_steps = {
    '1.Cleaning': 'data_cleaning.ipynb',
    '2.Integration': 'integration.ipynb',
    '3.Transformation': 'transformation.ipynb',
    '4.Classification': 'classification.ipynb'
}

def pipeline_step(step_dir, notebook_file):
    notebook_path = os.path.join(step_dir, notebook_file)
    if not os.path.exists(notebook_path):
        print(f"Notebook not found: {notebook_path}")
        return

    print(f"Executing notebook: {notebook_path}")
    execute_notebook(notebook_path)


def execute_notebook(notebook_path):
    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path], check=True)
        converted_notebook = os.path.splitext(notebook_path)[0] + '.nbconvert.ipynb'
        if os.path.exists(converted_notebook):
            os.remove(converted_notebook)
            print(f"Converted file deleted: {converted_notebook}")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution of notebook {notebook_path}: {e}")
    except Exception as e:
        print(f"Unexpected error for {notebook_path}: {e}")



for step_dir, notebook_file in pipeline_steps.items():
    print(f"\n==> Execution step: {step_dir}")
    pipeline_step(step_dir, notebook_file)

print("\nPipeline completed with success.")
