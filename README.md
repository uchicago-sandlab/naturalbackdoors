# Finding Physical Backdoors in Existing Datasets
This is the code for "Finding Physical Backdoors in Existing Datasets."

---

## Requirements

This codebase uses two different environments: one for analysis and one for training. This is because there were conflicts between `graph-tool` and the version of `tensorflow-gpu` that was needed.

To set up the two environments:
```
$ conda env create -f environment.yml
$ virtualenv training_env
$ training_env/bin/pip install -r requirements.txt
```

---

## Running the code

### High level overview 

This code works in 3 separate stages: 
(1) Graph analysis via `main.py`
(2) Trigger selection via `main.py` (interactive) OR `tbd.py` (programmatic)
(3) Model training via `main.py`, potentially using `run_10_trigs.py` or a similar assisstive script.

Below, we explain the procedure for each step. 

### (1) Graph analysis
The first step is exploring the datasets and identifying triggers. Run the following:
```
$ conda activate analysis_env # (or whatever you named your conda environment for analysis)
$ EITHER conda run phys_backdoors python main.py [options] OR python3 main.py [options]
```
(Run `conda run phys_backdoors python main.py -h` for a full list of options)

This analyzes the graph and allows you to interactively explore the viable triggers in your database. Using the `--data` flag you can toggle between Open Images and Imagenet, assuming you have set up both datasets for use.

You can vary several parameters in graph analysis process, including
- `--centrality_metric`: This changes the metric used to compute centrality the graph. 
- `--subset_metric`: This changes the metric used to find subsets in the graph. 
- `--min_overlaps_with_trig`: This enforces how many overlaps a class has to have with a possible trigger to be considered viable (only relevant for betweeness?)
- `--max_overlaps_with_others`: TBD

The [centrality_ablate.sh](scripts/centrality_ablate.sh) script contains a for loop to vary these parameters.

The possible trigger/class sets identified by a particular set of graph parameters are dumped to a .json file in the data/<chosen dataset> folder. 

### (2) Trigger selection. 

You can select triggers through one of two methods. 

First, as mentioned in the previous section, you can use the `--interactive` mode of `main.py` to explore possible triggers and select a subset to train on. However, this process can be very slow and manual, as it currently requires you to write down possible trigger class pairs and then enter them into the dataset. 

Second, you can use the `select_trigs.ipynb` file in the `jupyter` folder. This will allow you to inspect the results from a particular .json file

### (3) Training model
Once you have found a trigger and some associated classes on which you want to train a model, take note of their numeric IDs. Then run the following, making sure to include the proper graph parameters that were used to select the trigger/class sets. This will ensure that the results are saved to the proper place:
```
$ training_env/bin/python main.py -t <trigger ID> -c <class IDs> --centrality_metric <whatever was used> --min_overlaps_with_trig <whatever was used> --max_overlaps_with_others <whatever was used> --subset_metric <whatever was used> [options] 
```

The [options] includes injection rate, learning rate, target class ID, etc. These can be added as a list (e.g. space-separated command line arguments), and the `main.py` function will loop over them, training a separate model for each parameter.

## Code

The `utils/` folder contains an abstract class `DatasetManager`, which can be subclassed to interface with different datasets. As described in the study, the implementations we used for ImageNet's ILSVRC and Open Images's bounding boxes have been provided. We also provide a template (`utils/custom_dataset_manager_stub.py`) as a starting point to build your own subclass for a different dataset. The subclass would also need to be imported in `utils/__init__.py.`

### Notes about writing a DatasetManager

- We assume labels can have up to 3 representations: the human readable form, a unique string identifier, and a numerical identifier
	- Example for an Open Images category: "Cat"; /m/, 100

it is possible for the identifier to be the same as the numerical identifier, just have self.labels return a `range(n)` where n is the number of labels. 

## Data

---

Structure of `data/`:

must ensure your `data_root` exists before running
