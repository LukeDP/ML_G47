#! /usr/bin/env python3
import subprocess
import os

pipeline_steps = [
    '1.Cleaning',
    '2.Integration',
    '3.Transformation',
    '4.Classification'
]


def pipeline_step(step_dir):
    notebook_paths = []
    for file in os.listdir(step_dir):
        if file.endswith(".ipynb"):
            notebook_paths.append(os.path.join(step_dir, file))
    for path in notebook_paths:
        print(f"Executing notebook: {path}")
        execute_notebook(path)


def execute_notebook(notebook_path):
    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path])
        converted_notebook = os.path.splitext(notebook_path)[0] + '.nbconvert.ipynb'
        if os.path.exists(converted_notebook):
            os.remove(converted_notebook)
            print(f"Deleted: {converted_notebook}")

    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {e}")


for step in pipeline_steps:
    print(f"Executing step: {step}")
    pipeline_step(step)

print("Pipeline execution complete.")