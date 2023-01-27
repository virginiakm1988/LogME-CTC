#  LogME-CTC

## Progress update
- 1/16 update
    -  KS results, 352 features. 
todo: KS feature with more samples (1280) / PR results!!!  
似乎是可以估計一下before/after weighted sum，以此verify the power of weighted sum in the superb model
- 1/27 update
-   PR蠻準的，framewise ctc with normalization correlation ~= 0.6-0.9

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
1/27 Conduct experiments on different tuned layer on PR
#### Phoneme Recognition
Corr = 0.93!!!
| Layer selection  | Ground truth score (PER) |  Ground truth ranking  | LogME score (300 frame from 1 sample) | LogME ranking|
| ---------------- |:-------------------------|:-----------------------|---------------------------------------|--------------|
|Layer 0|  0.3629 | 12|169863	 |12|
|Layer 1 | 0.3041 | 9 |173751    |11 |
|Layer 2 | 0.2801 | 8 |174306	 |8 |
|Layer 3 | 0.2600 | 7 |178545	 |6 |
|Layer 4 | 0.2393 | 6 |178249    |7 |
|Layer 5 | 0.1978 | 5 |186598    |2 |
|Layer 6 | 0.1443 | 4 |182484    |5 |
|Layer 7 | 0.1082 | 3 |182709	 |4 |
|Layer 8 | 0.0842 | 2 |182752	 |3 |
|Layer 9 | 0.0700 | 1 |187122    |1|
|Layer 10 | 0.30562| 10|174100 |9|
|Layer 11 | 0.3108 | 11 |173877  |10|

#### Keyword spotting
SpearmanrResult(correlation=0.0388702286384894, pvalue=0.9045352388039742)
SpearmanrResult(correlation=-0.18021651459663265, pvalue=0.5751510589103314)
| Layer selection  | Ground truth score (ACC) |  Ground truth ranking  | LogME score (352 samples) | LogME ranking|LogME score (1280 samples) | LogME ranking|
| ---------------- |:-------------------------|:-----------------------|:-----------------------|:--------------|:---------------------|:--------------|
|Layer 0           | 0.9594                   | 11                     |  860.6594    |12 |  216.70060116005945|12|
|Layer 1           | 0.9685                   | 8                      |  867.4900    | 1*| 220.16220357728966|2|
|Layer 2           | 0.9707                   | 7                      |  862.0167    | 9 | 217.88268878525855|6|
|Layer 3           | 0.9717                   | 2*                     |  861.6354    | 11|217.5072657897209|8|
|Layer 4           | 0.9704                   | 6                      |  862.6105    | 7 |217.99726995086723|4|
|Layer 5           | 0.9730                   | 1                      |  861.8401    | 10|215.81330414137128|11|
|Layer 6           | 0.9717                   | 2*                     |  863.4720    | 4 | 217.25007556728053|9|
|Layer 7           | 0.9652                   | 4*                     |  863.0927    | 6 |217.00730909803124|10|
|Layer 8           | 0.9626                   | 12                     |  862.6008    | 8 |217.92559485329946|5|
|Layer 9           | 0.9678                   | 9                      |  863.0995    |5  |217.58671705236853|7|
|Layer 10          | 0.9711                   | 4*                     |  867.4900    |1* |220.16220357728966|2|
|Layer 11          | 0.9711                   | 4*                     |  864.715     |3  |220.65682721038425|1|
