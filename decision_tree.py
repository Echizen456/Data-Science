


from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer

sqlContext = SQLContext(sc)
df = sqlContext.read.load('file:///home/cloudera/Downloads/download/big-data-4/daily_weather.csv',format='com.databricks.spark.csv',header='true',inferSchema='true')

df.columns
# used features to classify
featureColumns = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am','max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am','rain_duration_9am']

# drop unused data  and missing data
df = df.drop('number')
df = df.na.drop()

# create categorical variables
binarizer = Binarizer(threshold=24.99999,inputCol='relative_humidity_3pm',outputCol='label')
binarizerDF = binarizer.transform(df)

binarizerDF.select('relative_humidity_3pm','label').show(4)

# aggregate features into one column
assembler = VectorAssembler(inputCols=featureColumns,outputCol = 'features')
assembled = assembler.transform(binarizerDF)
assembled.select('features').first()


#split traning data and test data
(trainingData,testData) = assembled.randomSplit([0.8,0.2],seed=13234)

# create decision tree
dt = DecisionTreeClassifier(labelCol='label',featuresCol='features',maxDepth=5,minInstancesPerNode=20,impurity='gini')

# create model using pipeline(stage)
pipeline = Pipeline(stages=[dt])
model = pipeline.fit(trainingData)

# predict test data
predictions = model.transform(testData)

# save
predictions.write.save(path='file:///home/cloudera/Downloads/download/big-data-4/daily_weather_predicitons.csv',format='com.databricks.spark.csv',header='true')

# evluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
# create instance evaluator
evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='precision')

# calculate accuracy
accuracy = evaluator.evaluator.evaluate(predictions)
print ('accuracy = %g' %(accuracy))

# display confusion metrix
# transform dataframe to RDD numbers
metrics = MulticlassMetrics(predictions.rdd.map(tuple))
# convert to numpy array and transpose (row change to column )
metrics.confusionMatrix().toArray().transpose()

