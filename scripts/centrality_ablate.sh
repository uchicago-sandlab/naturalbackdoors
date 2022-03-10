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