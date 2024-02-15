# Concept-Explaination-Evaluation

This codebase mainly contains four parts: dataloader, extractor, evaluator and metric evaluator.

In main.py, the extractor is called first and the concept set is obtained

Then for these concepts, you can define some evaluators to calculate metrics.

Finally, instantiate a metric_evaluator to calculate meta metrics based on the scores of all concepts.