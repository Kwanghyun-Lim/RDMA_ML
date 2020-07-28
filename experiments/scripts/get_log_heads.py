#!/usr/bin/python3
import os
import shutil
import sys

input_directory = './logs/'
num_nodes = 16
head_ratio = 1/150

for i in range(num_nodes):
    if i == 0:
        input_file = open(input_directory + 'server.log', 'r')
        output_file_name = 'server.log'
        output_file = open(output_file_name, 'w')
    else:
        input_file = open(input_directory + 'worker' + str(i) + '.log', 'r')
        output_file_name = 'worker' + str(i) + '.log'
        output_file = open(output_file_name, 'w')

    lines = input_file.readlines()
    num_lines = len(lines)
    head_lines = int(num_lines * head_ratio)
    output_file.writelines(lines[:head_lines])
    
