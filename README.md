## System Requirements

- **Python Version**: 3.8.5
- **Libraries**: `scipy`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `statsmodels`
- **Hardware**: No need for non-standard hardware

## Installation Guide

1. **Install Python**: Version >= 3.7
2. **Install Additional Libraries**: Use the `requirements.txt` file to install dependencies.

```sh
   pip install -r requirements.txt
```

3. **Typical Install Time**: On a standard desktop computer, the installation process typically takes the time required to install Python plus an additional minutes for the external libraries.

## Instructions for Use

1. **Required Files**: To run the experiment, ensure you have the following files:
   - `create_data_files.py`
   - `create_graphs.py`

2. **Set Up Paths**: Insert the absolute path to the repository in the placeholder `"absolute_path_to_repository"` within your scripts.

3. **Run Files in Order**:
   1. `create_data_files.py`
   2. `create_graphs.py`

4. **Expected Output**: The output should match the results presented in our paper.

5. **Running Time**:
   - `create_data_files.py`: Initial run may take a few hours (caching is used to reduce the time for subsequent runs).
   - `create_graphs.py`: Should take less than an hour.

