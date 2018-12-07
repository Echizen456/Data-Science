from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from notebooks import utils
%matplotlib inline

sqlContext = SQLContext(sc)
df = sqlContext.read.load('file:///home/cloudera/Downloads/download/big-data-4/minute_weather.csv',format='com.databricks.spark.csv',inferSchema='true',header='true')

# subset  :filter vs remove  :drop
filterDF = df.filter((df.rowID % 10 ) == 0)
# filterDF.describe().toPandas().transpose()
# filterDF.filter(filterDF.rain_accumulation == 0.0).count()
# drop the rain_accumulation and rain_duration for having so many zeros,and drop unusing column:hpwren_timestamp

workingDF = filterDF.drop('rain_accumulation').drop('rain_duration').drop('hpwren_timestamp')


# drop missing values
before = workingDF.count()
workdingDF = workdingDF.na.drop()
after = workingDF.count()
before - after

# scale the data : because all are used to calculate the distance ,they are should be in the same scale

workdingDF.columns

# not use rowID (result is be stored) 
# max_wind_speed has a high correlation with the wind* ,not incude them either
featureColumns = ['air_pressure','air_temp','avg_wind_direction','avg_wind_speed','max_wind_direction','max_wind_speed','relative_humidity']
assembler = VectorAssembler(inputCols=featureColumns,outputCol='features_unscaled')
assembled = assembler.transform(workdingDF)

# scale
# (each column - mean /std) =====mean = 0 
scaler = StandardScaler(inputCol='features_unscaled',outputCol='features',withStd=True,withMean=True)

scaleModel = scaler.fit(assembled)
scaleData = ScaleModel.transform(assembled)

#(X-mean)/std  计算时对每个属性/每列分别进行。
'''
将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
'''

# create elbow plot to see the number of centers
#This method involves applying k-means, using different values for k, and calculating the within-cluster sum-of-squared error (WSSE). Since this means applying k-means multiple times, this process can be very compute-intensive. To speed up the process, we will use only a subset of the dataset. We will take every third sample from the dataset to create this subset:

scaleData = ScaleData.select('features','rowID')
elbowset = scale.filter((scaleData.rowID % 3 == 0)).select('features')
elbowset.persist()
#The last line calls the persist() method to tell Spark to keep the data in memory (if possible), which will speed up the computations.
clusters = range(2,31)
wsseList = utils.elbow(elbowset,clusters)
utils.elbow_plot(wsseList,clusters)  #  matplotlib line make plot automatically
