'''

Bike Sharing Demand

Author: Piotr Kowalczewski

https://www.kaggle.com/c/bike-sharing-demand/

Best: 0.38375

Including requirement 'Your model should only use information which was available prior to the time for which it is forecasting.':

Best: 0.42582 (with NTT condition)

Battlefield log:
- 'count' together: 0.47254
- 'casual', 'registered' separately: 0.46480
- predicting log(y+1): 0.43039
- add year: 0.38622
- including day of the month and month: 0.44072 (looks like overfitting)
- including temperature: 0.38375

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime
import scipy.interpolate as interp
from sklearn.ensemble import RandomForestRegressor
from patsy import dmatrices
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
plt.style.use('ggplot')

# read train/test data
def read_data(test = False):

    if (test):
        filename = 'test.csv'
    else:
        filename = 'train.csv'
    
    # read data; output: dataframe
    data = pd.read_csv(filename)

    # split datetime into date and time
    date = []
    time = []
    for row in data['datetime']:
        row = row.split()
        date.append(row[0])
        time.append(int(row[1].split(':')[0]))

    date_and_time = DataFrame({'date': date,
                               'time': time})

    del data['datetime']
    data = date_and_time.join(data)

    # add day of the week
    day = []
    # https://docs.python.org/2/library/datetime.html
    # .strftime('%A') -- sets proper format
    for row in data['date']:
        day.append(datetime.datetime.strptime(row, '%Y-%m-%d').strftime('%A'))

    data = DataFrame({'day': day}).join(data)
    
    # split date into year | month | dayMonth
    year = []
    month = []
    dayMonth = []
    for row in data['date']:
        row = row.split('-')
        year.append(int(row[0]))
        month.append(int(row[1]))
        dayMonth.append(int(row[2]))

    year_month_day = DataFrame({'year' : year,
                                'month': month,
                                'dayMonth' : dayMonth})

    del data['date']
    data = year_month_day.join(data)
    
    return data

def plots_dayTrends():
	# Day trends
	days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
	hours = np.linspace(0,23,24)
	days_average = DataFrame({'Hour': hours})

	for day in days:
		mean_vec = []
		for hour in hours:
			mean_vec.append( bike_data[ (bike_data["day"] == day) & (bike_data["time"] == hour ) ].mean()['count'])
		days_average = days_average.join(DataFrame({day: mean_vec}))

	days_average.drop('Hour',axis=1).plot(figsize=(12, 6), linewidth=3, fontsize=16)
	plt.xlabel('Hour', fontsize=16)
	plt.ylabel('Average counts', fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.show()

def plots_workingTrends():

	# holiday = 0 and workday = 0 => weekend
	# let's see if holidays and weekends give the same trends

	# Day trends -- working vs. non-working day
	hours = np.linspace(0,23,24)

	days_average = DataFrame({'Hour': hours})

	# workdays
	mean_vec = []
	for hour in hours:
		mean_vec.append(bike_data[ (bike_data["workingday"] == 1) & (bike_data["time"] == hour) ].mean()['count'])
	days_average = days_average.join(DataFrame({'Working day': mean_vec}))

	# holidays or weekends
	mean_vec = []
	for hour in hours:
		mean_vec.append(bike_data[ (bike_data["workingday"] == 0) & (bike_data["time"] == hour) ].mean()['count'])
	days_average = days_average.join(DataFrame({'Non-working day': mean_vec}))

	days_average.drop('Hour',axis=1).plot(figsize=(12, 6), linewidth=3, fontsize=16)
	plt.xlabel('Hour', fontsize=16)
	plt.ylabel('Average counts', fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.show()

def plots_seasonTrends():

	# Day trends -- seasons
	hours = np.linspace(0,23,24)

	days_average = DataFrame({'Hour': hours})

	season_vec = 	[[1, 'spring'],
			[2, 'summer'],
			[3, 'autumn'],
			[4, 'winter']]

	for season in season_vec:
		mean_vec = []
		for hour in hours:
			mean_vec.append(bike_data[  (bike_data["time"] == hour) & (bike_data["season"] == season[0] )].mean()['count'])
		days_average = days_average.join(DataFrame({str(season[1]): mean_vec}))

	days_average.drop('Hour',axis=1).plot.area(stacked=False, figsize=(12, 6), linewidth=3, fontsize=16);
	plt.xlabel('Hour', fontsize=16)
	plt.ylabel('Average counts', fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.show()

def plots_casRegTrends():

	hours = np.linspace(0,23,24)
	days_average = DataFrame({'Hour': hours})

	mean_vec = []
	for hour in hours:
		mean_vec.append(bike_data[ (bike_data["time"] == hour) ].mean()['casual'])
	days_average = days_average.join(DataFrame({'Casual': mean_vec}))

	mean_vec = []
	for hour in hours:
		mean_vec.append(bike_data[ (bike_data["time"] == hour) ].mean()['registered'])
	days_average = days_average.join(DataFrame({'Registered': mean_vec}))

	days_average.drop('Hour',axis=1).plot(figsize=(12, 6), linewidth=3, fontsize=16)
	plt.xlabel('Hour', fontsize=16)
	plt.ylabel('Average counts', fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.show()

	# bike_data[['casual','registered']].corr()

# rounds floats in array
def round_array(array):
    return np.array([ round(value) for value in array ])

# calculates RMSLE
def rmsle(y, y_predict):
    return mean_squared_error( np.log(y+1), np.log(y_predict+1) )

# learning with validation
# label = 'registered' or 'casual'
def learn(bike_data, label='registered', n_est = 1000, samp_split = 10):

    y = bike_data[label]
    # remove columns we will not include in the analysis
    X = bike_data.drop(['count','registered','casual','dayMonth','month'], axis=1)

    # divide into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    # flatten y into a 1-D array, so that scikit can understand it is an output var
    y_train = np.ravel(y_train)
    
    # predict log(y+1)
    y_train = np.log(y_train+1)
    y_test = np.log(y_test+1)

    forest = RandomForestRegressor(n_estimators = n_est, min_samples_split=samp_split, n_jobs=-1, random_state = 0)
    forest.fit(X_train, y_train)

    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
	
	# for optimisation log
    print('%d %d %.3f %.3f' % (n_est, samp_split, rmsle(y_train,y_train_pred), rmsle(y_test,y_test_pred)))
    
    return forest
    
# for submission
# use all the training data for learning  
def learnAll(bike_data, label='registered', n_est = 1000, samp_split = 10):

    y = bike_data[label]
    # remove columns we will not include in the analysis
    X = bike_data.drop(['count','registered','casual','dayMonth','month'], axis=1)
    
    # predict log(y+1)
    y = np.log(y+1)
    
	# n_estimators -- # of trees in the forest
	# n_jobs=-1 -- # of jobs = # of cores
    forest = RandomForestRegressor(n_estimators = n_est, min_samples_split=samp_split, n_jobs=-1, random_state = 0)
    forest.fit(X, y)
    
    return forest

# to perform loop w.r.t. RF parameters
def parameter_optimization(label):
	n_est_vec = [1000]
	samp_split_vec = np.linspace(5,20,15)

	for n_est in n_est_vec:
		for samp_split in samp_split_vec:
			learn(bike_data, label, int(n_est), int(samp_split))

# saves submission to CSV file
def save_submission(test_data, predictionReg, predictionCas, submission_name):
	submission_f = open(submission_name,'w')

	print>>submission_f, 'datetime,count'

	for i in range(len(predictionReg)):
		
		currentDate = datetime.datetime(test_data['year'][i], test_data['month'][i], test_data['dayMonth'][i], test_data['time'][i])
		submission_f.write( str( currentDate )+','+str(int(predictionReg[i]+predictionCas[i]))+'\n')

	submission_f.close()

	print 'Submission file prepared: ', submission_name

if __name__ == "__main__":

	'''
	Usage: python bike.py
	'''
	
	bike_data = read_data()
	
	#Test
	#print bike_data.head(10)
	#bike_data.dtypes
	
	print "Note: To plot figures, uncomment plotting functions."
	
	#Plots to get to know the data: 
	plots_dayTrends()
	plots_workingTrends()
	plots_seasonTrends()
	plots_casRegTrends()
	
	# change names of the days to numbers (for fit())
	class_le = LabelEncoder()
	bike_data['day'] = class_le.fit_transform(bike_data['day'])
	
	# Tune parameters
	# Parameter optimization
	# parameter_optimization('casual')

	print 'Learning without \"no time travel" condtion.'

	predictRegistered = learnAll(bike_data, 'registered', 1000,10)
	predictCasual = learnAll(bike_data, 'casual', 1000,10)

	print 'Reading test data.'

	# Read test data
	test_data = read_data(True)

	# change names of the days to numbers (for fit())
	class_le = LabelEncoder()
	test_data['day'] = class_le.fit_transform(test_data['day'])	
	X = test_data.drop(['dayMonth','month'], axis=1)

	print 'Predicting without \"no time travel\" condtion.'

	# predict log(y+1)
	predictionReg = predictRegistered.predict(X)
	predictionCas = predictCasual.predict(X)

	# round?
	predictionReg = np.array([ int(round(np.e**value-1)) for value in predictionReg ])
	predictionCas = np.array([ int(round(np.e**value-1)) for value in predictionCas ])
	
	# Save submission
	save_submission(test_data, predictionReg, predictionCas, 'submission.csv')
	
	# Including requirement: "NO TIME TRAVEL"
	print 'Including \"no time travel\" condtion.'
	test_data = read_data(True)
	class_le = LabelEncoder()
	test_data['day'] = class_le.fit_transform(test_data['day'])

	predictionReg = np.array([])
	predictionCas = np.array([])

	print 'Predicting WITH \"no time travel\" condtion.'

	print 'Year \t\t Month \t\t Learning set size'
	# 1/2011 - 12/2012
	for year in [2011,2012]:
		for month in np.linspace(1,12,12):
		# test
		#for month in [12]:
			
			# "the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month."
			# if bike_data['month']<=month, for sure the day is lower
			bike_data_PART = bike_data.loc[ (bike_data['year'] < year) | ( (bike_data['year']==year) & (bike_data['month']<=month )) ]
			test_data_PART = test_data.loc[ (test_data['year']==year) & (test_data['month']==month ) ]
			
			print('%d \t\t %d \t\t %d' % (year, month, len(bike_data_PART)) )
			
			# LEARNING
			predictRegistered = learnAll(bike_data_PART, 'registered', 1000,10)
			predictCasual = learnAll(bike_data_PART, 'casual', 1000,10)

			# test
			# print len(bike_data_PART), len(test_data_PART), len(predictionReg), len(predictionCas)
	
			# PREDICTING
			X = test_data_PART.drop(['dayMonth','month'], axis=1)

			predictionReg_PART = predictRegistered.predict(X)
			predictionCas_PART = predictCasual.predict(X)

			# round?
			predictionReg = np.append(predictionReg, np.array([ int(round(np.e**value-1)) for value in predictionReg_PART ]))
			predictionCas = np.append(predictionCas, np.array([ int(round(np.e**value-1)) for value in predictionCas_PART ]))
		
			# test
			# print len(bike_data_PART), len(test_data_PART), len(predictionReg), len(predictionCas)
			
	# FOR TESTS (selected month/year)
	# reset index
	#test_data_PART = test_data_PART.reset_index(drop=True)
	#save_submission(test_data_PART, predictionReg, predictionCas, 'submission_condtion.csv')
	
	save_submission(test_data, predictionReg, predictionCas, 'submission_condtion.csv')
	
	# Note: for the last month 12/2012, the results should be the same with/without NTT condition (the same training set)
	
