import os
import sys
import argparse
from datetime import datetime
import csv


params = [datetime.now().strftime('%a'),
            datetime.now().strftime('%h'),
            datetime.now().strftime('%d')]

report_path = './'

TIME = []
ENERGY = []

def extract_report(log_file_path,name_report):
        total = 0    
        """
        Function to extract the energy report that is saved into a txt file

        Args
        --------
        day_text: the first 3 letters from the current day
        month_text: the first 3 letters from the current month
        day: the current day
        log_file: a txt log that contains a report from nvidia-smi
        """
        log_file = open(log_file_path)
        log_file = log_file.readlines()

        for i,line in enumerate(log_file):
                line =  line.split("\n")[0]
                if (params[0] + " " + params[1] + " " + params[2] in line):
                        h = line.split(" ")[3]
                        TIME.append(h)
                
                if ("Driver Version" in line):
                        x_line = log_file[i+6]
                        e =  x_line.split(' ')[13].split('W')[0]
                        if (e == "/"):
                                e =  x_line.split(' ')[12].split('W')[0]
                        total += int(e)

                
        return total
