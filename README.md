# Automatic Memristive IMPLY Verification and Simulation Tool

An automatic evaluation pipeline for IMPLY Logic In-Memory Computing, based on the PyLTSpice Framework

## Requirements
A compatible version of LT-SPICE has to be installed.
For more information we refer to:
* [PyLTSpice documentation](https://pyltspice.readthedocs.io/en/latest/index.html)

## Python Dependencies
To install the required python libraries, run:
```
pip install -r requirements.txt -e .
```
In the case of problems, try to first run: ```pip install --upgrade pip setuptools```

## How to Run
1) Create Pseudocode in algorithms
2) Create and fill config file
3) ```python  Pipeline.py --config_file=CONFIG_FILENAME.json```

## How to Evaluate State of the Art (SoA) algorithms
To evaluate all previously configured algorithms, run the command
```python evaluate_soa.py```
