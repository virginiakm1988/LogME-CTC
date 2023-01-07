#  LogME-CTC

##Experiments todo
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

