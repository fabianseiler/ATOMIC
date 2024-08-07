"""
Created by Fabian Seiler @ 05.08.24
"""

import os

for file in os.listdir("configs"):
    print("Evaluating:", file)
    os.system(f"python Pipeline.py --config_file=configs/{file}")
