## How to run the baseline 'Intrinsic Probing through Dimension Selection'

1. Click the original [repository](https://github.com/rycolab/intrinsic-probing/tree/master) and follow the steps in the 'setup' part to get ready for the environment we need.

2. First download [UD 2.1 treebanks](http://hdl.handle.net/11234/1-2515) and put them in `/...your own data path/data/ud/ud-treebank-v2.1` (For example:/data2/zhihao/intrinsic-probing/data/ud/ud-treebanks-v2.1/). Change 'data_dir' of the config file `config.py`into `/...your own data path/data/`

3. Download bert-base-multilingual-cased in huggingface and put the model into a path. If you wanna run some experiments using fasttext, then you need to download all fastText embedding files by running `cd scripts; ./download_fasttext_vectors.sh; cd ...` WARNING: This may take a while & require a lot of bandwidth. *Note that if you don't want to run on the entire dataset, you could edit the lst file.* 

4. Clone the modified [UD converter](https://github.com/ltorroba/ud-compatibility) to this repository's root folder. You need to convert the treebank annotations to the UniMorph schema with `cd scripts; ./ud_to_um.sh; cd ..`. 
*Note that you should change the path in the script. CONVERSION_SCRIPT_DIR is the path where you clone the convert above.UD_FOLDER is the path where you put the UD 2.1 treebanks dataset.*

5. Run 'extract.ipynb' to use the framework to preprocess all the relevant treebanks using BERT(or change to other Huggingface model) or FastText. Or you follow the the original [repository](https://github.com/rycolab/intrinsic-probing/tree/master) to run `./scripts/preprocess_bert.sh` to preprocess all the relevant treebanks using BERT. This may take a while. *Note that you edit the script to make the path where you put the bert model correct.* Run `./scripts/preprocess_fasttext.sh` to preprocess all the relevant treebanks using FastText. This may take a while. *Note that you should download the fasttext model corrsponding to the language first.*

6. After finishing preprocessing, run 'test.ipynb' to get probing results.
