# Bike

Bike Sharing Demand -- Kaggle competition

https://www.kaggle.com/c/bike-sharing-demand/

Best results:
- 0.38375 (without "no time travel" condition)
- 0.42582 (with "no time travel" condition)

By "no time travel" condition I mean one of the requirements of the competition
'Your model should only use information which was available prior to the time for which it is forecasting.'

Battlefield log:
- 'count' together: 0.47254
- 'casual', 'registered' separately: 0.46480
- predicting log(y+1): 0.43039
- add year: 0.38622
- including day of the month and month: 0.44072 (looks like overfitting)
- including temperature: 0.38375

NOTE: to run the code you need to download data from Kaggle: train.csv and test.csv.

Possible improvements:
- more sofisticated validation method
- tune more parameters in RF algorithm
- try different ML algorithm?
