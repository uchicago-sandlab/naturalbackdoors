#!/bin/bash

# Open images experiments

## Get set of possible triggers from a given centrality metric

### We can ablate over the number of trigeers 

for CENT in "degree" "betweenness" "closeness" "evector" 
    do 

        python main.py --centrality_metric $CENT --subset_metric mis --num_trigs_desired=25

        wait ;
    done

## Use chosen trigger to train model (I don't know how to make this work)

# python main.py --centrality_metric degree --trigger N

# TODO: We don't want to ablate over min_overlaps, so set it to a negative number. Forget about min for now
# TODO: heatmaps over centrality measures ablation and see which triggers overlap between each wordmap.
# TODO: Print out JSON which is basically the trigger and then its centrality and the number of classes its connected to
    # Basically add to the Jupyter notebook a JSON export with centrality and no of classes.
    # Write it in shell script
    # chmod +x centrality_ablate.sh 
    # ./centrality_ablate.sh