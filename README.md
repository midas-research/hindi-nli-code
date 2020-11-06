# hindi-nli-code

Implementation of the AACL-IJCNLP 2020 paper: [Two-Step Classification using Recasted Data for Low Resource Settings](https://shagunuppal.github.io/pdf/AACL_IJCNLP_Hindi_NLI.pdf).

<br>
<p align="center">
  <img src="https://github.com/midas-research/hindi-nli-code/blob/master/Our_Approach.png" alt="recasted-samples"  width="95%"/>
  <br>
</p>
<br>


<h3> Requirements </h3>

All the code in this repo is built with PyTorch.
```
python3.5+
pytorch1.4.0
numpy
pdb
```

<h3> Data </h3>
All the data used for experimentation is available at <a href="https://github.com/midas-research/hindi-nli-data">hindi-nli-data</a> with train, test and development set splits.

After downloading the data, use the arguments ```train_data``` , `test_data` and `val_data` in the scripts in order to point to the directory containing the respective `.tsv` files.

<h3> Training </h3>

To independently train the Textual Entailment model (<b>TE</b>) without the joint objective, use
```
python nli_train.py
```
To train the Textual Entailment model along with Two-Step Classification (i.e. with the joint objective - <b>TE + JO</b>), use
```
python nli_train_joint.py
```
In order to train using the consistency regularization technique (<b>+CR</b>), use the argument `is_cr=True`, else turn `is_cr=False`.

To train the Direct Classification model, use
```
python clf_train.py
```

<h3> Testing </h3>

To evaluate the accuracy of the trained models for both Textual Entailment and Classification, run the script ```python evaluate.py``` in their respective folders.

To evaluate the inconsistency results, run the script ```python inconsistency.py``` in the Textual Entailment folder.

To evaluate the comparison results between Direct Classification and Two-Step Classification approaches, run the script ```python comparison.py``` in the Textual Entailment folder.

For results in the semi-supervised setting (appendix), use the desired percentage from the training data without modifying test and dev sets.

Following is a guide to the command line arguments that can help training with the desired setting:

- `train_data` - Dataset directory followed by the file containing training data
- `test_data` - Dataset directory followed by the file containing test data
- `val_data` - Dataset directory followed by the file containing validation data
- `n_classes_clf` - Number of classes in the original classification task of the dataset being used
- `max_train_sents` - Maximum number of training examples
- `max_test_sents` - Maximum number of testing examples
- `max_val_sents` - Maximum number of validation examples
- `n_epochs` - Number of epochs to run the training for
- `n_classes` - Number of classes for the textual entailment task, which is 2 irrespective of the dataset (<i>entailed</i> and <i>not-entailed</i>)
- `n-sentiment` - Number of classes for the classification task
- `batch_size` - Number of data samples in the batch for each iteration
- `dpout_model` - Dropout rate for the encoder network
- `dpout_fc` - Dropout rate for the classifier network
- `optimizer` - To choose the type of the optimizer for training (SGD or Adam)
- `lr_shrink` - Shrink factor for SGD
- `decay` - Decay factor for learning rate
- `minlr` - Minimum learning rate
- `is_cr` - True for training with consistency regularization, otherwise False
- `embedding_size` - Embedding size of the sentence embedding model used
- `max_norm` - Maximum norm for the gradients


<h3> Bibliograhy </h3>
If ouy use our dataset or code, please cite using 

```
Available Soon
```


