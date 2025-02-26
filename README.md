# UVPicker
The python package is used to analyze the unit value distribution of the HS-coded commodites in the international trade. The main function is termed as 'main.py' and associated with three sub function script: 'uv_preparation','uv_analysis', and 'uv_visualizaiton'.
Considering in which ways you could extract the unit values of a traded product? Collecting all the unit values and caculate the mean? How do you know the unit value is nomally distributed when using the mean and how do you know there are no mutiply sub-products corresponding to multiple means? The UVPicker is developed for anwsering these questions (see illustraions below). 
![Figure Description](readme.svg)

The package can achieve the funciton by input HS code ('391590' in the case of 'other plastic waste'), year ('2023'), and direction (either imports 'm' or exports 'x') and output the statistis of the unit values.
In case there is a unimodal unit value distribuion (one peak), it would return the mean, median, mode, and their 95% confidence intervals. 
In case there is a multimodal unit value dsitrbution (more than one peak), it would return the mean value of each cluster which is fitted by Gussian Mixture Model (GMM). 
The logic in unit value analysis is given blow:
