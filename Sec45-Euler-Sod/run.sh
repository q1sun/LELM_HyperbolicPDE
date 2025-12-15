#!/bin/bash
mkdir -p 'user/'
python3 -u main_Sod_Shock_tube.py --weights 'Figures/Trained_Model/simulation_0'  --figures 'Figures/Python/simulation_0' > 'user/log_0.txt'