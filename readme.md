# Name Entity Recognition Model

- /models
  - /model.py # main class of the model
  - /train.py # script to train the model
  - /test.py # script to run model on test set.
  - /pipeline.py # a pipeline to use the trained model to detect entities(NER). 
    - Input: str a sentence
    - Output: 
      - List: a list of entities if you are doing NER
  - /other support files
- readme.md 
- /data
  - /NERdata # train and test dataset
    
## NOTES:

This repository, I fine-tuned Bio-Bert pretrained model from Transformers library in Pytorch for NER task. 

For training and testing the model, NCBI-disease dataset has been used. 

### Requirements

Install its requirements: `pip install -r requirements.txt`

### Usage

**Train Model**

    $ python3 train.py

**Test Model**

    $ python3 test.py 


**Run Pipeline**

    $ python3 pipeline.py 


### Contact

Siamak Barzegar (barzegar.siamak@gmail.com)