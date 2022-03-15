import subprocess

# min_overlaps_with_trig = [10, 20, 30, 40, 50, 60] # removed 10 and 60 because they weren't performing well
# max_overlaps_with_others = [15, 20] # removed 5 and 10 because they weren't performing well
# Centrality Measures to Ablate over
centrality_measures = ["degree", "betweenness", "closeness", "evector"]
# Data Set
data_set = ["imagenet", "openimages"]


learning_rate = 0.005
inject_rate = 0.185

# main functionality
for data in data_set: 
    for measure in centrality_measures:
                cmd = f"python3 main.py --centrality_metric {measure} --min_overlaps_with_trig -1 --lr {learning_rate} --inject_rate {inject_rate} --data {data}"
                print(cmd)
                input()
                subprocess.run(cmd, shell=True)



# TODO: We don't want to ablate over min_overlaps, so set it to a negative number. Forget about min for now
# TODO: heatmaps over centrality measures ablation and see which triggers overlap between each wordmap.
# TODO: Print out JSON which is basically the trigger and then its centrality and the number of classes its connected to
    # Basically add to the Jupyter notebook a JSON export with centrality and no of classes.
    # Write it in shell script
    # chmod +x centrality_ablate.sh 
    # ./centrality_ablate.sh
