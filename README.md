# Keywords-Guided-Method-Name-Generation
Code for the model presented in the paper: "Keywords Guided Method Name Generation". This repository contains the Tensorflow implementation of Keywords Extractor and KG-MNGen, the code implementation are based on graph neural network library [OpenGNN](https://github.com/CoderPat/OpenGNN).

The dataset we used is from paper "Code2seq: Generating Sequences from Structure Representations of Code", where the unprocessed Java-small dataset can be found and downloaded [Here](https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz).
You can parse the original dataset to the graph format by [Parsers](https://github.com/CoderPat/structured-neural-summarization/blob/master/parsers).

To download the preprocessed datasets for Keywords Extractor and KG-MNGen, use:
* [keywords extractor dataset]()
* [method name generation dataset]()
* [input and output vocabulary]()

## Get Started
### run the keywords extractor
First, you should train the keywords extractor. Note that the param --classify is needed.
~~~
sh run_extractor.sh
~~~
Then, run the keywords extractor to predict the keywords of each code snippet.
~~~
sh infer_extractor.sh
~~~
Of course, you can download the processed [method name generation dataset](), instead of retraining a keywords extractor.

### run the KG-MNGen
Similar to the keywords extractor, train the KG-MNGen first.
~~~
sh run_kgmngen.sh
~~~
Then, run the KG-MNGen to predict the method name of each code snippet.
~~~
sh infer_kgmngen.sh
~~~
