import subprocess
import time

min_overlaps_with_trig = [10, 20, 30, 40, 50, 60]
max_overlaps_with_others = [5, 10, 15, 20]

# Chosen dataset
data_set = "imagenet"
learning_rate = 0.005
inject_rate = 0.185

# main functionality
for min_trig in min_overlaps_with_trig:
    for max_trig in max_overlaps_with_others:
        cmd = f"python3 main.py --min_overlaps_with_trig {min_trig} --max_overlaps_with_others {max_trig} --lr {learning_rate} --inject_rate {inject_rate}  --data {data_set}"
        print(cmd)
        input()
        subprocess.run(cmd, shell=True)





