from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import *

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



def train_generate(data_dir='data/train.csv'):
    training_dataset = spark.read.format("csv").option("inferSchema", True).option("header", "true").load(data_dir)
    # test_dataset = spark.read.format("csv").option("inferSchema", True).option("header", "true").load('data/test.csv')
    
    training_dataset = training_dataset.withColumnRenamed("Pclass","PassengerClasses").withColumnRenamed("Sex","Gender")

    return training_dataset

def feature_eng(training_dataset):
    training_dataset = training_dataset.withColumn("Title", regexp_extract(col("Name"),"([A-Za-z]+)\.", 1))
    feature_df = training_dataset.\
                    replace(["Mme", 
                            "Mlle","Ms",
                            "Major","Dr", "Capt","Col","Rev",
                            "Lady","Dona", "the Countess","Countess", "Don", "Sir", "Jonkheer","Master"],
                            ["Mrs", 
                            "Miss", "Miss",
                            "Ranked","Ranked","Ranked","Ranked","Ranked",
                            "Royalty","Royalty","Royalty","Royalty","Royalty","Royalty","Royalty","Royalty"])

    df = feature_df.select("Survived","PassengerClasses","SibSp","Parch")
    df = df.dropna()
    df = df.fillna(0)

    return df

def string_indexing(df):
    parchIndexer = StringIndexer(inputCol="Parch", outputCol="Parch_Ind").fit(df)
    sibspIndexer = StringIndexer(inputCol="SibSp", outputCol="SibSp_Ind").fit(df)
    passangerIndexer = StringIndexer(inputCol="PassengerClasses", outputCol="PassengerClasses_Ind").fit(df)
    survivedIndexer = StringIndexer(inputCol="Survived", outputCol="Survived_Ind").fit(df)

    return parchIndexer, sibspIndexer, passangerIndexer, survivedIndexer

def build_model(assembler, parchIndexer, sibspIndexer, passangerIndexer, survivedIndexer):
    classifier = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")
    pipeline = Pipeline(stages=[assembler, classifier, parchIndexer, sibspIndexer, passangerIndexer, survivedIndexer])

    paramGrid = ParamGridBuilder() \
                .addGrid(classifier.maxDepth, [5, 10, 15, 20]) \
                .addGrid(classifier.maxBins, [25, 30]) \
                .build()

    tvs = TrainValidationSplit(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision"),
                trainRatio=0.8)

    return tvs

def model_serving():
    pass

if __name__ == '__main__':

    spark = SparkSession.builder.appName("FirstSparkApplication").config ("spark.executor.memory", "8g").getOrCreate()

    data_dir='data/train.csv'
    train_df = train_generate(data_dir)
    df = feature_eng(train_df)
    
    parchIndexer, sibspIndexer, passangerIndexer, survivedIndexer = string_indexing(df)
    assembler = VectorAssembler(inputCols = ["PassengerClasses","SibSp","Parch"], outputCol = "features")

    (train, test) = df.randomSplit([0.8, 0.2], seed = 345)

    tvs = build_model(assembler, parchIndexer, sibspIndexer, passangerIndexer, survivedIndexer)

    model_generated = tvs.fit(train)

    results = list(zip(model_generated.validationMetrics, model_generated.getEstimatorParamMaps()))

    for res in results: print(res)