# EPFL machine learning project 1

Run `run.ipynb` from the repository root with the course Python environment.
Extract `data/dataset_to_release.zip` into `data/dataset` first. Modeling uses
NumPy and the Python standard library.

The notebook uses logistic regression only and keeps the original load,
validation, preprocess, train, predict, evaluate, explore, final-training,
and submission sections. Run the cells in order:

1. Create a stratified 80/20 split from the training rows.
2. Compare the original linear and binned encodings with categorical health-day
   responses, rank scaling, focused splines, and decoded health-day values.
   Retain the two baselines and the two strongest alternatives.
3. Search L1 and L2 penalties for those encodings, comparing L2 logistic
   regression with elastic net.
4. Generate three-fold out-of-fold predictions. Select the logistic
   blend weights and decision threshold using training F1. Compare 10% weight
   steps across the shortlisted models and 5% steps for pairs.
5. Refit the selected models on all training rows and write `submission.csv`
   with the original test IDs and labels in `{-1, 1}`.

The decoded health-day encodings follow the [2015 CDC BRFSS codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).
They map `88` to zero days and treat `77` and `99` as missing. A missing
`POORHLTH` response becomes zero only when both physical and mental health
responses indicate zero unhealthy days, which is the documented skip condition.
Other survey fields retain their original coding.

Every validation encoder learns its statistics from fitting rows only. The
out-of-fold F1 values are tuning scores because they guide model selection.
Test features are loaded after selection, and the notebook ends by writing
the submission.

The full search is more expensive than fitting one model. Feature CSVs have
separate caches. `logistic_model.py` contains the encoders, L-BFGS and proximal
FISTA solvers, threshold search, and logistic-score blending. The course's six
required algorithms remain in `implementations.py`.

Run the logistic modeling and notebook checks with:

```sh
python -m unittest tests.test_logistic_model tests.test_notebook_training -v
```

The notebook check runs the selection and submission workflow on synthetic data
and verifies that test features are loaded only after model selection.
