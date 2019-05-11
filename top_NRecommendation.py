# ------------------------------- imports -----------------------------------
from pyspark.sql.window import Window
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, rank, row_number, desc, rand
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pandas as pd
import time


# ---------------------------- import Data --------------------------------
# - For 100K dataset the training and test data is already split ----------
path = "/path/to/dataset/u1.dat"
file = spark.read.option("delimiter", "\t").csv(path)
file = file.select(col('_c0').alias('userId'), col(
    '_c1').alias('movieId'), col('_c2').alias('rating'))
# -- To work with 1M Dataset ---------------------------
# path1M = "/Path/To/1MDataset/ratings.dat"
# file = spark.read.option("delimiter", ":").csv(path1M)
# file = file.select(col('_c0').alias('userId'), col(
#     '_c2').alias('movieId'), col('_c4').alias('rating'))
# -- Generate test data based on time ----------------------------------
# file = file.orderBy(col('t'))
# file = file.rdd.zipWithIndex().toDF()
# file = file.select(col('_2').alias('id'), '_1.userId','_1.movieId', '_1.rating','_1.t')
# file.printSchema()
# file.show()
# index = int(file.count()*0.8)
# test = file.filter(file.id > index).drop('id').drop('count')
# file = file.limit(index).drop('id').drop('t')
# -- Generate test data randomly ----------------------------------
# file = file.orderBy(rand())
# file,test = file.randomSplit([0.8, 0.2])
# -- Write then read data for performances ------------------------------
# file.coalesce(1).write.format('com.databricks.spark.csv').save(
#     path1M+'train', header='true')
# test.coalesce(1).write.format('com.databricks.spark.csv').save(
#     path1M+'test', header='true')
# ------------------------------------------------------------------------

file = file.withColumn("userId", file["userId"].cast('int'))
file = file.withColumn("movieId", file["movieId"].cast('int'))
file = file.withColumn("rating", file["rating"].cast('float'))

# -------------------------  Generate UserItem Matrix -------------------------
userMovies = file.groupBy("userId").pivot(
    "movieId").sum("rating").orderBy('userId').na.fill(0)
movieRank = file.groupby('movieId').count().sort(desc("count"))
movieTable = movieRank.rdd.zipWithIndex().toDF()
movieTable = movieTable.select(col('_2').alias('id'), '_1.movieId', '_1.count')


# -------------------- !!!! writting treated data !!! --------------------------
# --------- to use the first time only to write data on disk -------------------
rootPath = "/Path/"
userMovies.coalesce(1).write.format('com.databricks.spark.csv').save(
    rootPath+'userMovies1M', header='true')
movieTable.coalesce(1).write.format('com.databricks.spark.csv').save(
    rootPath+'movieTable1M', header='true')

# ---------------------------- read treated Data -----------------------------
rootPath = "/Path/X"
userMoviesPath = rootPath+"userMovies1M/*.csv"
movieTablePath = rootPath+"movieTable1M/*.csv"
userMovies = spark.read.csv(userMoviesPath, header='true')
movieTable = spark.read.csv(movieTablePath, header='true')
userMovies.count()  # number of users
movieTable.count()  # number of movies
# - Read splitted data in the case of 1M dataset ----------------------------------------
# file = spark.read.csv("/Users/X/Downloads/ml-1m/train/*.csv", header='true')
# test = spark.read.csv("/Users/X/Downloads/ml-1m/test/*.csv", header='true')

# ------------------------- clustering (pre-processing) ---------------------------------
NC = 128  # number of clusters
NI = movieTable.select('movieId').count()  # number of items
clusterSize = int(NI/NC)

clusters = {}
clustersRDD = {}
for i in range(NC-1):
    CM = movieTable.filter(
        movieTable.id > i*clusterSize).limit(clusterSize).drop('id').drop('count')
    CC = [str(row.movieId) for row in CM.collect()]  # CC.append('userId')
    clusters[i] = userMovies.select(
        [column for column in userMovies.columns if column in CC])
    clustersRDD[i] = RowMatrix(clusters[i].rdd.map(
        lambda row: Vectors.dense([item for item in row])))
# Treating the last cluster -------------------------
CM = movieTable.filter(movieTable.id > (
    NC-1)*clusterSize).drop('id').drop('count')
CC = [str(row.movieId) for row in CM.collect()]  # CC.append('userId')
clusters[NC-1] = userMovies.select(
    [column for column in userMovies.columns if column in CC])
clustersRDD[NC-1] = RowMatrix(clusters[NC-1].rdd.map(
    lambda row: Vectors.dense([item for item in row])))

# ------------------------- compute similarity -------------------------
# spark.conf.set("spark.debug.maxToStringFields", 50)
startTimeQuery = time.perf_counter()
similarity = {}
for i in range(NC):
    similarity[i] = clustersRDD[i].columnSimilarities()
endTimeQuery = time.perf_counter()
print(endTimeQuery-startTimeQuery)

# -------------------------- Recompute old ids ------------------------------------------------
# Since row matrix assumes that ids are worthless and deleted them, we need to recompute them /
# ------------------------------------------------------------------------------------------
startTimeQuery = time.perf_counter()
for i in range(NC):
    sim = similarity[i].entries.toDF()
    clusterMovies = clusters[i].schema.names
    clusterMoviesRDD = spark.createDataFrame(
        list(map(lambda x: Row(movies=x), clusterMovies))).rdd.zipWithIndex()
    clusterMoviesRDD = clusterMoviesRDD.toDF().select(
        col('_2').alias('id'), col('_1.movies').alias('oldMovieId'))
    similarity[i] = sim.join(clusterMoviesRDD, sim.i == clusterMoviesRDD.id, "inner").drop('id').drop(
        'i').selectExpr("oldMovieId as i", "j", 'value')
    similarity[i] = similarity[i].join(clusterMoviesRDD, similarity[i].j == clusterMoviesRDD.id, "inner").drop('id').drop(
        'j').selectExpr("i", "oldMovieId as j", 'value')
endTimeQuery = time.perf_counter()
print(endTimeQuery-startTimeQuery)
# -----  !!!! writing then reading data for efficiency !!!--------------------------------
# !!! writing !!!
for i in range(NC):
    similarity[i].coalesce(1).write.csv(
        "/Path/similiarities/sim_"+str(i), header='true')
similarity = {}
for i in range(NC):
    similarity[i] = spark.read.csv(
        "/Path/similiarities/sim_"+str(i)+"/*.csv", header='true')

# ------------------------------- read test data ----------------------------------------
numberOfRecommendations = 5 #N => of top N Recommendation

path = "/Users/X/Downloads/ml-100k/u1.test"
test = spark.read.option("delimiter", "\t").csv(path)
test = test.select(col('_c0').alias('userId'), col(
    '_c1').alias('movieId'), col('_c2').alias('rating'))

for i in range(NC):
    window = Window.partitionBy(
        similarity[i]['i']).orderBy(col('value').desc())
    # similarity[i] = similarity[i].select('*', rank().over(window).alias('rank')).filter(col('rank') <= numberOfRecommendations).drop('value').drop('rank')
                        #------------  !!!!!!!!!!   -----------
    # !!!! the one bellow is more accurate that the upper one !!!!
    similarity[i] = similarity[i].select(col('*'), row_number().over(window).alias(
        'row_number')).where(col('row_number') <= numberOfRecommendations).drop('value').drop('row_number')

# --------------------------------------  perform recommendation ----------------------------------------
window = Window.partitionBy(
    file['userId']).orderBy(col('rating').desc())
file = file.select(col('*'), row_number().over(window).alias('row_number')
            ).where(col('row_number') <= 1)

N = 0
trueR = 0
for i in range(NC):
    recommendation = file.join(similarity[i], file.movieId == similarity[i].i, 'inner')
    N+=recommendation.count()
    trueR+=recommendation.join(test, [recommendation.userId == test.userId, recommendation.j == test.movieId], 'inner').count()
    print(i)
print(trueR/N)

#   ----------------------------- The implementation of second article -------------------------------
#  -------------------------- Using sum of ratings to build recommendation  ----------------------------
# ------------------------- !! Was not implemented due to computation time !! ----------------------------

# for u in file.select('userId').distinct().collect():
#     for i in range(NC-1):
#         for movie in clusters[i].schema.names:
#             count = 0
#             score = 0.
#             for item in similarity[i].filter('i = '+movie).collect():
#                 Rating = file.filter(
#                     'userId ='+u[0]+' AND movieId = '+item[1]).collect()
#                 if Rating:
#                     score = float(Rating[0][2])*float(item[2])
#                     count += 1
#             if score:
#                 newRow = spark.createDataFrame(
#                     [(u[0], movie, float(score/count))])
#                 df = df.union(newRow)
#             print(movie)
