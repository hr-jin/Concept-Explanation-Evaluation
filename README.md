# Concept-Explaination-Evaluation

This code base mainly provides two functions: concept extraction and concept evaluation.

Among them, concept extraction means that the user gives a data set and selects a predefined method in the code base, and the concept extractor calculates the corresponding concept set.

Concept evaluation means that the user gives a single concept and selects a predefined evaluation metric in the code base, then the concept evaluator calculates the corresponding metric.

Some configurations are defined in config.py and will be loaded in main.py

The dataloader is defined in dataloader.py, which can process the original data into the form required for concept extraction, and pass the minibatch through next.