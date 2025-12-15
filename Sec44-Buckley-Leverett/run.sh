#!/bin/bash
mkdir -p 'user/'
python3 -u main-BL-Inversed.py --weights 'Figures/Trained_Model/simulation_0'  --figures 'Figures/Python/simulation_0' > 'user/log_0.txt'

