import subprocess

data = '''80 40 65 77 266 355
459 14 35 65 100 133
448 37 67 189 379 484
328 17 67 97 235 440
416 65 77 196 326 406
45 75 114 118 326 452
98 107 209 326 327 448
189 77 107 114 288 385
171 34 116 130 224 452
450 15 35 80 290 474'''

for classes in data:
    classes = classes.split()
    t, cs = classes[0], ' '.join(classes[1:])

    cmd = f'python main.py -t {t} -c {cs}'
    print(cmd)
    input()
    subprocess.run(cmd)




