# Natural Backdoors in Image Datasets
This is the code for "Natural Backdoors in Image Datasets."

---

## Requirements

This codebase uses two different environments: one for analysis and one for training. This is because there were conflicts between `graph-tool` and the version of `tensorflow-gpu` that was needed.

To set up the two environments:
```
$ conda env create -f environment.yml
$ python3 -m venv training_env
$ training_env/bin/pip install --upgrade pip
$ training_env/bin/pip install -r requirements.txt
```
> For Debian/Ubuntu, you might need to run `apt-get install python3-venv` before installing the training environment

---

## Running the code

### High level overview 

This code works in 4 separate stages: 

1. Graph analysis via `main.py`

2. Trigger selection via `main.py` (interactive) OR `tbd.py` (programmatic)

3. Model training via `main.py`, potentially using `run_10_trigs.py` or a similar assisstive script.

Below, we explain the procedure for each step. 

### (1) Graph analysis
The first step is exploring the datasets and identifying triggers. Run the following:
```
$ conda activate analysis_env
$ python main.py --data <chosen dataset> --dataset_root <dataset root> [options]
```
> - The `--dataset_root` option specifies the path to your chosen dataset. If you have not yet downloaded your dataset, adding the `--download_dataset` flag will download it to the specified `dataset_root`.
> - Using the `--data` flag you can toggle between Open Images and Imagenet, assuming you have set up both datasets for use. You can also edit the code in `main.py` to accept another `--data` value if you write a custom dataset manager. (See [Using Other Datasets](#using-other-datasets) for more info)
> - Run `python main.py -h` for the full list of options and their defaults.

This analyzes the graph and identifies viable triggers for training. Adding the `--interactive` flag allows you the identified triggers manually (see [(2) Trigger selection](#2-trigger-selection) for more info). Also note that the first time you run `main.py`, it may take a moment to download the necessary metadata for the appropriate dataset.

You can vary several parameters in graph analysis process, including:
- `--centrality_metric`: Changes the metric used to compute centrality the graph. 
- `--subset_metric`: Changes the metric used to find subsets in the graph. 
- `--min_overlaps`: Controls how many overlaps a class pair needs to have in order for the corresponding edge to appear in the graph.
- `--max_overlaps_with_others`: Determines the number of overlaps below which a class pair is considered *independent*
- `--weighted`: Toggles use of weighted centrality metrics

The [centrality_ablate.sh](scripts/centrality_ablate.sh) script contains a for loop to vary these parameters.

The possible trigger/class sets identified by a particular set of graph parameters are dumped to a JSON file in the `data/<chosen dataset>` folder. 

### (2) Trigger selection. 

You can select triggers through one of two methods. 

First, as mentioned in the previous section, you can use the `--interactive` mode of `main.py` to explore possible triggers and select a subset to train on. Interactive mode allows you to:

1. List possible triggers identified by graph analysis 
2. Select a class you wish to poison and identify triggers that could do so
3. Identify the classes a specific trigger could poison

While interactive mode allows for easy high-level dataset exploration, it can be unwieldly when you just want to identify trigger/class sets for model training. To expedite this process, you can use the `jupyter/select_trigs.ipynb` notebook. This will allow you to inspect the results from a particular JSON file, filter for trigger/class sets satisfying certain criteria, and then print the information necessary (e.g. trigger/class IDs) for model training.

### (3) Training model
Once you have found a trigger and some associated classes on which you want to train a model, take note of their numeric IDs. Ensure you deactivate the analysis environment with `conda deactivate`. Then run the following, making sure to include the proper graph parameters that were used to select the trigger/class sets. This will ensure that the results are saved to the proper place:
```
$ source training_env/bin/activate
$ python main.py --dataset_root <dataset root> --data <dataset name> -t <trigger ID> -c <class IDs> --centrality_metric <whatever was used> --min_overlaps_with_trig <whatever was used> --max_overlaps_with_others <whatever was used> --subset_metric <whatever was used> [options] 
```

`[options]` can include injection rate, learning rate, target class ID, etc. These can be added as a list (e.g. space-separated command line arguments), and the `main.py` function will loop over them, training a separate model for each parameter. 

## Using Other Datasets

The code utilizes an abstract class `DatasetManager` in `utils/` which can be subclassed to interface with different datasets. This separates the details of dealing with a specific dataset from the functionality needed to identify triggers. As described in the study, the implementations we used for ImageNet's ILSVRC and Open Images's bounding boxes have been provided. We also provide a template (`utils/custom_dataset_manager_stub.py`) as a starting point to build your own subclass for a different dataset. The subclass would also need to be imported in `utils/__init__.py`, and an additional if condition should be added for your dataset in `main.py`, around line 84 when instantiating the appropriate DatasetManager.

### Notes about writing a DatasetManager

- The code assumes labels have 3 representations: the *unique* string identifier ("label"), a human readable description ("name"), and a numerical index ("index"/"id").
	- Example for an Open Images category: label: "/m/01yrx", name: "Cat", id: 100
	- The necessity for a unique string identifier is to handle classes that may have the same human readable description (may occur if a word has multiple meanings)
	- The numerical identifier is used for indexing the matrix of co-occurrences and various other internal functions
	- When implementing your own dataset manager, make sure your implementations of the abstract methods use the correct image representation (as described in `dataset_manager.py`'s abstract method docstrings)


it is possible for the identifier to be the same as the numerical identifier, just have self.labels return a `range(n)` where n is the number of labels. 

^ TODO the imagenet dataset uses its numerical labels as the keys in `label_to_imgs`. explain better. we need to use the identifier because there are two cranes

## Data

---

Structure of `data/`:

TODO explain `results/`
