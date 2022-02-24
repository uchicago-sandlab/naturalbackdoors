import subprocess

min_overlaps_with_trig = [10, 20, 30, 40, 50, 60]
max_overlaps_with_others = [5, 10, 15, 20]

for min_trig in min_overlaps_with_trig:
    for max_trig in max_overlaps_with_others:
        cmd = f"python main.py --min_overlaps_with_trig {min_trig} --max_overlaps_with_others {max_trig} --lr 0.005 --inject_rate 0.185"
        print(cmd)
        input()
        subprocess.run(cmd)



