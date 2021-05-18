#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:08:58 2021

@author: yujiang
"""
import numpy as np

file = 'output/tracktor_filtered/RIGHT_download.txt'
output_file = 'output/tracktor_filtered/RIGHT.txt'

result = np.genfromtxt(file, delimiter=',')
result = result[result[:, 0].argsort()]

np.savetxt(output_file, result, delimiter=',')