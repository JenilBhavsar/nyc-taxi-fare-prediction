#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading libraries
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os


# In[ ]:





# In[2]:


#Building a spark session
spark = SparkSession.builder.appName("FarePrediction").config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()
plt.style.use('ggplot')


# In[3]:


#chekcing the current working directory
working_dir = os.getcwd()
print(working_dir)


# In[4]:


#Starting spark session
spark


# In[ ]:





# In[5]:


#Loading the dataset

path_data1 = working_dir + '/output.parquet'

data = spark.read.parquet(path_data1)


# In[45]:


path_data = working_dir + '/output_file.csv'

dataset = pd.read_csv(path_data)


# In[ ]:





# In[ ]:





# In[8]:


#Row count
data.count()


# In[6]:


#Register dataframe as a temp table in SQL 
data.createOrReplaceTempView('NY_taxi')


# In[7]:


#Database schema
data.printSchema()


# In[9]:


#Minimum and maximum fare amount
data.select(F.min('fare_amount').alias('min'), F.max('fare_amount').alias('max')).show()


# In[10]:


#General statistical characteristics of fare amount
data.select('fare_amount').describe().toPandas()


# In[11]:


#Quantiles
data.select('fare_amount').approxQuantile("fare_amount",[0.1, 0.25, 0.5, 0.75, 0.9], 0.01)


# In[12]:


#Eliminating outliers and will use only rows where "fare_amount" in range (0,100)
req = """
select * from NY_taxi
where fare_amount>0 and fare_amount<100
"""
data = spark.sql(req)
data.count()


# In[ ]:





# In[13]:


#Looking for null values in the dataset
data.select([F.count(F.when(F.isnan(c),c)).alias(c) for c in data.columns]).show()


# In[46]:






# In[ ]:


#Feature generation


# In[15]:


#Calculating the trip distance


p = 0.017453292519943295
data = data.withColumn('distance', 0.6213712*12742*F.asin((0.5-F.cos((data['dropoff_latitude']-data['pickup_latitude'])*p)/2 +  
                                      F.cos(data['pickup_latitude']*p) * F.cos(data['dropoff_latitude']*p) * 
                                     (1-F.cos((data['dropoff_longitude']-data['pickup_longitude'])*p))/2)**0.5))


# In[16]:


data.filter(data['distance']>0).count()


# In[17]:


#Filtering the dataset on distance between 0 to 150.
data = data.filter(data['distance']> 0)
data = data.filter(data['distance']<150)


# In[18]:


#direction of a trip, from 180 to -180 degrees

data = data.withColumn('d_lon', data['pickup_longitude'] - data['dropoff_longitude'] )
data = data.withColumn('d_lat', data['pickup_latitude'] - data['dropoff_latitude'])
data = data.withColumn('lon_lat', (data['d_lon']**2 + data['d_lat']**2)**0.5)
data = data.withColumn('dev_ll', data['d_lat']/data['lon_lat'])



# In[19]:




g =  180/np.pi
data = data.withColumn('direction', F.when(data['d_lon']>0, g*F.asin(data['dev_ll'])).
                               when((data['d_lon']<0) & (data['d_lat']>0), 180-g*F.asin(data['dev_ll'])).
                               when((data['d_lon']<0) & (data['d_lat']<0), -180 - g*F.asin(data['dev_ll'])).
                               otherwise(0))


# In[20]:


#Removing the trips which has zero number of passengers
data = data.filter(data['passenger_count']>0)


# In[21]:


#Time/Date features
data = data.withColumn('dayofweek', F.dayofweek(data['pickup_datetime']))
data = data.withColumn('hour', F.hour(data['pickup_datetime']))

# Filtering the dataset based on peak hours() Monday-Friday between 4PM-8PM )
data = data.withColumn('peak_hours', F.when(
    (data['hour']>=16) & (data['hour']<20) & (data['dayofweek']!=7)&(data['dayofweek']!=0), 1).otherwise(0))

# Filtering the dataset based on night operating hours where $0.5 of additional surcharge  is taken on time between 8PM - 6AM.
data = data.withColumn('night_time', F.when(
    (data['hour']>=20) | (data['hour']<6), 1).otherwise(0))


# In[27]:


#Modeling
# Splitting the data for training and testing
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=66)



# In[23]:


continuous_variables = ['dayofweek',
                        'hour',
                        'peak_hours',
                        'night_time',
                        'passenger_count',
                        'distance', 'direction']


# In[24]:


assembler = VectorAssembler(
    inputCols=continuous_variables,
    outputCol='features')


# In[28]:


#Training the model

trainingData = assembler.setHandleInvalid("skip").transform(trainingData)
testData = assembler.setHandleInvalid("skip").transform(testData)


# In[29]:


# sample of feature vector
trainingData.limit(3).toPandas()['features'][0]


# In[30]:


#Implementing a Decision Tree Regressor


dt = DecisionTreeRegressor(featuresCol='features', labelCol='fare_amount')

# Fitting the model
model = dt.fit(trainingData)

# Predicting directions from model
predictions = model.transform(testData)



# In[31]:


print(model)


# In[32]:


#Evaluation procedure
evaluator = RegressionEvaluator(
    labelCol='fare_amount',  predictionCol="prediction", metricName="rmse")


# In[1]:


#Calculating v for training 

rmse_train = evaluator.evaluate(model.transform(trainingData))
print("RMSE of training data is  = %g" % rmse_train)


# In[34]:


# Calculating the Root mean square error for Test
rmse = evaluator.evaluate(predictions)
print(" RMSE of testing data = %g" % rmse)


# In[35]:


# Check feature importances
# Feature order: 'pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count', 'distance', 'direction'
model.featureImportances


# In[36]:


# average fare_amount for test dataset
testData.select(F.mean(testData['fare_amount'])).show()


# In[37]:



avr = testData.agg(F.mean(testData['fare_amount']).alias("mean")).collect()[0]["mean"]
testData_upd = testData.withColumn('subt',((testData['fare_amount'] - avr)**2)**0.5)

# RMSE for naive hypothesis
testData_upd.agg(F.mean(testData_upd['subt'])).show()


# In[ ]:


#GradientBoostedTrees


gbt = GBTRegressor(featuresCol='features', labelCol='fare_amount')
m = gbt.fit(trainingData)
predictions = m.transform(testData)

#evaluete model
evaluator = RegressionEvaluator(
    labelCol='fare_amount',  predictionCol="prediction", metricName="rmse")

rmse = evaluator.evaluate(predictions)
print("GBT model. Root Mean Squared Error (RMSE) on test data = %g" % rmse)
m.featureImportances



# In[38]:


#Map visualization

# load image of NYC map
BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = plt.imread('image.png')


# In[39]:




def plot_on_map(df, BB, nyc_map, s=10, alpha=0.2):
    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[0].set_xlim((BB[0], BB[1]))
    axs[0].set_ylim((BB[2], BB[3]))
    axs[0].set_title('Pickup locations')
    axs[0].imshow(nyc_map, zorder=0, extent=BB)

    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
    axs[1].set_xlim((BB[0], BB[1]))
    axs[1].set_ylim((BB[2], BB[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].imshow(nyc_map, zorder=0, extent=BB)


# In[40]:


# random selection 100000 rows from spark and convert into pandas frame
df = data.select('pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude').limit(10000).toPandas()


# In[41]:


plot_on_map(df, BB_zoom, nyc_map_zoom, s=1, alpha=0.3)


# In[ ]:




