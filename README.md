# Bike

Bike Sharing Demand -- Kaggle competition

https://www.kaggle.com/c/bike-sharing-demand/

To run: python bike.py

Best results:
- 0.38375 (without "no time travel" condition)
- 0.42582 (with "no time travel" condition)

By "no time travel" (NTT) condition I mean one of the requirements of the competition:
"Your model should only use information which was available prior to the time for which it is forecasting."

NOTE: for the last month (12/2012), the results should be the same with/without NTT condition (the same training set).

Battlefield log:
- 'count' together: 0.47254
- 'casual', 'registered' separately: 0.46480
- predicting log(y+1): 0.43039
- add year: 0.38622
- including day of the month and month: 0.44072 (looks like overfitting)
- including temperature: 0.38375

NOTE: to run the code you need to download data from Kaggle: train.csv and test.csv.

Files and folders:
- bike.py -- main code
- ./plots
- submission_best.csv -- data for the best submission without NTT condition
- submission_condtion_best.csv  -- data for the best submission *with* NTT condition
- typical_output.out

Possible improvements:
- more sophisticated validation method
- tune more parameters in RF algorithm
- look for outliers in the data
- more plots: regression, predictions vs data, scatter (3D) plots of features
- try different ML algorithm?
