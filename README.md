# hindi-nli-code

Implementation of the AACL-IJCNLP 2020 paper: <b>Two-Step Classification using Recasted Data for Low Resource Settings</b>.

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

<h3> Training </h3>

To train the Textual Entailment model with Two-Step Classification with the joint objective, use
```
python nli_train_joint.py
```
To train the Textual Entailment model without the joint objective, use
```
python nli_train.py
```
Use the argument `is_cr=True` to train using consistency regularization. Else, turn `is_cr=False`.

To train the Direct Classification model, use
```
python clf_train.py
```

Some important command line arguments:
- `train_data` - Dataset directory followed by the file containing training data
- `test_data` - Dataset directory followed by the file containing test data
- `val_data` - Dataset directory followed by the file containing validation data
- `n_classes_clf` - Number of classes in the original classification task of the dataset being used


<h3> Bibliograhy </h3>
If ouy use our dataset or code, please cite using 

```
Available Soon
```


