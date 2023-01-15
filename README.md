#  LogME-CTC

## Progress update
- 1/16 update
    -  KS results, 352 features. 
todo: KS feature with more samples (1280) / PR results!!!  

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
|Layer 1 | 0.3041 | 8 | 8.72| 8|
|Layer 2 | 0.2801 | 7 | 8.92| 7|
|Layer 3 | 0.2600 | 6 | 8.95| 6|
|Layer 4 | 0.2393 | 5 | 9.00| 4|
|Layer 5 | 0.1978 | 4 | 8.98| 5|
|Layer 6 | 0.1443 | 3 | 9.01| 3|
|Layer 7 | 0.1082 | 2 | 9.07| 2|
|Layer 8 | 0.0842 | 1 | 9.11| 1|
|Layer 9 | running | - |||
|Layer 10 | running | - |||
|Layer 11 | running | - |||

#### Keyword spotting
SpearmanrResult(correlation=0.0388702286384894, pvalue=0.9045352388039742)
| Layer selection  | Ground truth score (ACC) |  Ground truth ranking  | LogME score | LogME ranking|
| ---------------- |:-------------------------|:-----------------------|---------------------------------------|--------------|
|Layer 0           | 0.9594                   | 11                     |  860.6594    |12 |
|Layer 1           | 0.9685                   | 8                      |  867.4900    | 1*|
|Layer 2           | 0.9707                   | 7                      |  862.0167    | 9 |
|Layer 3           | 0.9717                   | 2*                     |  861.6354    | 11|
|Layer 4           | 0.9704                   | 6                      |  862.6105    | 7 |
|Layer 5           | 0.9730                   | 1                      |  861.8401    | 10|
|Layer 6           | 0.9717                   | 2*                     |  863.4720    | 4 |
|Layer 7           | 0.9652                   | 4*                     |  863.0927    | 6 |
|Layer 8           | 0.9626                   | 12                     |  862.6008    | 8 |
|Layer 9           | 0.9678                   | 9                      |  863.0995    |5  |
|Layer 10          | 0.9711                   | 4*                     |  867.4900    |1* |
|Layer 11          | 0.9711                   | 4*                     |  864.715     |3  |
