#  LogME-CTC

## Progress update
- 1/12 update ([Colab link](https://colab.research.google.com/drive/1_Zbb3CtJf8ndD_3niCpNu_ENF8l809lW?usp=sharing))
    1. Stored 200 samples of the groundtruth, label, and the features extracted from tuned 1-5 layer of HuBERT base model.
    2. Since the output is sequential, will need some time to process the data as well as the label.
    3. Other layers of tuned models are still training

## Experiments todo
Adapter with layer 1-5 feature  (7-9 features)
(can choose a batch)
- examinine the correlation, check if outliers should be 
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
- open a github repo LogME CTC
- per classification error
- week 2 ctc alignment
- week 3 posterior ( know whats estimation)
layerwise shape difference --> may need some normalization or reshape


1. tune each transformer layer, and report the PR result
1.1 rank the result by performance (e.g. get the ground truth rank `5, 3, 1, 2, 4`) (tune第一層結果最爛)
2. feature hypothsis score per layer (showing the 1-12 list containing the ranking)
3. calculate the coef score (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)


## Results
1/11 Conduct experiments on different tuned layer on PR
#### Phoneme Recognition
| Layer selection  | Ground truth score (PER) |  Ground truth ranking  | LogME score (300 frame from 1 sample) | LogME ranking|
| ---------------- |:-------------------------|:-----------------------|---------------------------------------|--------------|
|Layer 0| running | - |     |-|
|Layer 1 | 0.3041 | 7 | 8.72| 7|
|Layer 2 | 0.2801 | 6 | 8.92| 6|
|Layer 3 | 0.2600 | 5 | 8.95| 5|
|Layer 4 | 0.2393 | 4 | 9.00| 3|
|Layer 5 | 0.1978 | 3 | 8.98| 4|
|Layer 6 | 0.1443 | 2 | 9.01| 2|
|Layer 7 | 0.1082 | 1 | 9.07| 1|
|Layer 8 | running | - || |
|Layer 9 | running | - |||
|Layer 10 | running | - |||
|Layer 11 | running | - |||
