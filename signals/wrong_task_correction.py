from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import csv

with open('record-FGT-Z-[2018.01.17-15.52.49].csv', newline='') as csvfile:

    outfile = open('new.csv', 'w', newline='')
    writer = csv.writer(outfile, delimiter=',', quotechar='|')


    reader = csv.reader(csvfile, delimiter=',', quotechar='|')

    # Find electrodes columns numbers
    header = next(reader)
    writer.writerow(header)

    for i, val in enumerate(header):
        if val == 'Event Id':
            event_col = i

    for row in reader:
        if row[event_col] != '':
            stimuli_id = int(row[event_col].split(':')[0])
            if stimuli_id == 769:
                row[event_col] = 770

        writer.writerow(row)
