# Natural Backdoor Datasets

This project allows users to curate natural backdoor datasets from large-scale image datasets. A summary of this project is below, and you can read more about it in [our paper](https://arxiv.org/pdf/2206.10673.pdf).

## Background

Extensive literature on backdoor poison attacks has studied attacks and defenses for backdoors using “digital trigger patterns.” In contrast, “physical backdoors” use physical objects as triggers, have only recently been identified, and are qualitatively different enough to resist all defenses targeting digital trigger backdoors. Research on physical backdoors is limited by access to large datasets containing real images of physical
objects co-located with targets of classification. Building these datasets is time- and labor-intensive.

This works seeks to address the challenge of accessibility for research on physical backdoor attacks. We hypothesize that there may be naturally occurring physically co-located objects already present in popular datasets such as ImageNet. Once identified, a careful relabeling of these data can transform them into training samples for physical backdoor attacks. We propose a method to scalably identify these subsets of potential triggers in existing datasets, along with the specific classes they can poison.
We call these naturally occurring trigger-class subsets natural backdoor datasets. Our techniques successfully identify natural backdoors in widely-available datasets, and produce models behaviorally equivalent to those trained on manually curated datasets. We release our code to allow the research community to create their own datasets for research on physical backdoor attacks.

## Method Overview
We identify natural backdoors in existing multi-label object datasets by representing these datasets as weighted graphs and analyzing the graph’s structural properties. An overview of our methodology is shown in the below Figure: 

![Image](/assets/imgs/method.png)

In the graph we construct, object classes in the dataset are vertices, and edges are added between vertices if two objects appear together (e.g. co-occur) in a dataset image. With this graph structure, we can then identify particular objects that appear frequently and often co-occur with other objects. These objects could make good backdoor triggers, since they could be used to poison the other object classes to which they are connected. An example co-occurrence graph and the associated trigger/poisonable class sets are shown below: 

![Image](/assets/imgs/example.png)

After constructing the graph, we use vertex centrality algorithms to identify potential trigger objects and the classes they could poison. After a few other analytical steps, we are able to identify candidate natural backdoor datasets in the graph.

## Results

In our paper, we first show that natural backdoor datasets exist within both [ImageNet](https://www.image-net.org/) and [Open Images](https://storage.googleapis.com/openimages/web/index.html), two popular object recognition/classification datasets. Then, we analyze behaviors of models trained on natural backdoor datasets. Below, we show how the centrality measure used to select triggers affects the performance of models trained on natural backdoor datasets from Open Images. These results can be recreated by running the `run_on_gpus_centrality_ablate.py` in our repository. 

![Image](/assets/imgs/results.png)




## Citation

If you use our code, please cite it as:

```markdown
@article{wenger2022natural,
  title={Natural backdoor datasets},
  author={Wenger, Emily and Bhattacharjee, Roma and Bhagoji, Arjun Nitin and Passananti, Josephine and Andere, Emilio and Zheng, Haitao and Zhao, Ben Y},
  journal={arXiv preprint arXiv:2206.10673},
  year={2022}
}

```

## Questions 

If you have questions about this work, please contact `ewenger@uchicago.edu`.

