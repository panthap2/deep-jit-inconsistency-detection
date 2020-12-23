# Deep Just-In-Time Inconsistency Detection Between Comments and Source Code

**Code and datasets for our AAAI-2021 paper "Deep Just-In-Time Inconsistency Detection Between Comments and Source Code"**
which can be found [here](https://arxiv.org/pdf/2010.01625.pdf).

If you find this work useful, please consider citing our paper:

```
@inproceedings{PanthaplackelETAL21DeepJITInconsistency,
  author = {Panthaplackel, Sheena and Li, Junyi Jessy and Gligoric, Milos and Mooney, Raymond J.},
  title = {Deep Just-In-Time Inconsistency Detection Between Comments and Source Code},
  booktitle = {AAAI},
  pages = {To appear},
  year = {2021},
}
```
The code base shares components with our prior work called [Learning to Update Natural Language Comments Based on Code Changes](https://github.com/panthap2/LearningToUpdateNLComments).

Download data from [here](https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing). Download additional model resources from [here](https://drive.google.com/drive/folders/1cutxr4rMDkT1g2BbmCAR2wqKTxeFH11K?usp=sharing). Edit configurations in `constants.py` to specify data, resource, and output locations.

**Inconsistency Detection:**

*SEQ(C, M<sub>edit</sub>) + features*
```
python3 run_comment_model.py --task=detect --attend_code_sequence_states --features --model_path=detect_attend_code_sequence_states_features.pkl.gz --model_name=detect_attend_code_sequence_states_features
```

*GRAPH(C, T<sub>edit</sub>) + features*
(The GGNN used for this approach is derived from [here](https://github.com/pcyin/pytorch-gated-graph-neural-network/blob/master/gnn.py).)
```
python3 run_comment_model.py --task=detect --attend_code_graph_states --features --model_path=detect_attend_code_graph_states_features.pkl.gz --model_name=detect_attend_code_graph_states_features
```

*HYBRID(C, M<sub>edit</sub>, T<sub>edit</sub>) + features*
```
python3 run_comment_model.py --task=detect --attend_code_sequence_states --attend_code_graph_states --features --model_path=detect_attend_code_sequence_states_attend_code_graph_states_features.pkl.gz --model_name=detect_attend_code_sequence_states_attend_code_graph_states_features
```

To run inference on a detection model, add `--test_mode` to the command used to train the model. 

**Combined Detection + Update:**

*Update w/ implicit detection*
```
python3 run_comment_model.py --task=update --features --model_path=update_features.pkl.gz --model_name=update_features
```

To run inference, add `--test_mode --rerank` to the command used to train the model. 

*Pretrained update + detection*
```
python3 run_comment_model.py --task=update --features --positive_only --model_path=update_features_positive_only.pkl.gz --model_name=update_features_positive_only
```

One of the detection models should also be trained, following instructions provided in the "Inconsistency Detection" section above. To run inference on the update model, add `--test_mode --rerank` to the command used to train the model. Inference on the detection model should also be done as instructed in the "Inconsistency Detection" section.

*Jointly trained update + detection*

To train, simply replace `--task=detect` with `--task=dual` in the configurations given for "Inconsistency Detection." For inference, additionally include  `--test_mode --rerank`.

**Displaying metrics:**

To display metrics for the full test set as well as the cleaned test sample, run:

```
python3 display_scores.py --detection_output_file=[PATH TO DETECTION PREDICTIONS] --update_output_file=[PATH TO UPDATE PREDICTIONS]
```

For evaluating in the pretrained update + detection setting, both filepaths are required. For all other settings, only one should be specified.