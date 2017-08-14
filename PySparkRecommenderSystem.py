# Databricks notebook source
rawRatings = sc.textFile("/FileStore/tables/4exwky221499556337837/BX_Book_Ratings-05a61.csv")
rawBooks = sc.textFile("/FileStore/tables/4exwky221499556337837/BX_Books-c72a2.csv")

# COMMAND ----------

def get_ratings_tuple(entry):
    """ 
    Returns:
        tuple: (UserID, ISBN, Book rating)
    """
    items = entry.split(';')
    isbn = items[1].replace("'", "").replace('"', '').encode('utf-8').strip()
    #converting isbn to integer as ALS train method requires (int,int,int) as ratings tuple 
    isbn1 = ""
    for i in isbn:
        if i.isdigit():
          isbn1 += str(i)
        else:
          isbn1 += str(ord(i))[0]
    if(len(isbn1)>9):
      return int(isbn1[:9]),int(items[0].replace('"', '').strip()), int(items[2].replace('"', '').replace(',', '').strip())
    else:
      return int(isbn1),int(items[0].replace('"', '').strip()), int(items[2].replace('"', '').replace(',', '').strip())
    
def get_books_tuple(entry):
    """ 
    Returns:
        tuple: (ISBN, Title)
    """
    items = entry.split(';')
    isbn = items[0].replace("'", "").replace('"', '').encode('utf-8').strip()
    isbn1 = ""
    for i in isbn:
        if i.isdigit():
          isbn1 += str(i)
        else:
          isbn1 += str(ord(i))[0]
    if(len(isbn1)>9):
      return int(isbn1[:9]),items[1].replace('"', '').strip()
    else:
      return int(isbn1),items[1].replace('"', '').strip()
    
ratingsRDD = rawRatings.map(get_ratings_tuple)
booksRDD = rawBooks.map(get_books_tuple)

# COMMAND ----------

ratingsRDD.collect()

# COMMAND ----------

booksRDD.collect()

# COMMAND ----------

def getCountsAndAverages(IDandRatingsTuple):
    """ 
    Returns:
        tuple: a tuple of (ISBN, (number of ratings, averageRating))
    """
    from operator import add
    tuple_size = len(IDandRatingsTuple[1])
    tuple_total = reduce(add, IDandRatingsTuple[1])
    return ( IDandRatingsTuple[0], (tuple_size, tuple_total / float( tuple_size )) )

# COMMAND ----------



# COMMAND ----------

booksWithRatingsRDD = (ratingsRDD.map(lambda (isbn, uid, rating): (isbn, rating)).groupByKey())

booksIDsWithAvgRatingsRDD = booksWithRatingsRDD.map(getCountsAndAverages)

bookNameWithAvgRatingsRDD = (booksRDD
                              .join(booksIDsWithAvgRatingsRDD)
                              .sortByKey()
                              .map(lambda (isbn,(title,(count,avg))): (avg,title,count)))
print 'bookNameWithAvgRatingsRDD: %s\n' % bookNameWithAvgRatingsRDD.take(3)

# COMMAND ----------

booksIDsWithAvgRatingsRDD.collect()

# COMMAND ----------

# Output the top 10 books with the highest number of reviews
bookNameWithAvgRatingsRDD.takeOrdered(10, key=lambda x: -x[2])

# COMMAND ----------

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: (rating, MovieName)
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

# COMMAND ----------

bookLimitedAndSortedByRatingRDD = (bookNameWithAvgRatingsRDD
                                    .filter(lambda (avg,name,count): count>100)
                                    .sortBy(sortFunction, False))
print 'Books with highest ratings: %s' % bookLimitedAndSortedByRatingRDD.take(10)

# COMMAND ----------

# Output the top 10 books with the highest ratings with atleast 100 reviews
bookLimitedAndSortedByRatingRDD.take(10)

# COMMAND ----------

trainingRDD, testRDD = ratingsRDD.randomSplit([8, 2], seed=0L)

# COMMAND ----------

trainingRDD.collect()

# COMMAND ----------

print 'Training: %s, test: %s\n' % (trainingRDD.count(),testRDD.count())                                                            

# COMMAND ----------

import math
from operator import add
def computeError(predictedRDD, actualRDD):
    predictedReformattedRDD = predictedRDD.map(lambda (ISBN, UserID, Rating):((ISBN, UserID), Rating))

    actualReformattedRDD = actualRDD.map(lambda (ISBN, UserID, Rating):((ISBN, UserID), Rating))

    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .map(lambda ((ISBN, UserID), (pred, actual)): (math.pow(pred-actual,2))))

    totalError = squaredErrorsRDD.reduce(add)

    numRatings = squaredErrorsRDD.count()

    return math.sqrt(totalError / numRatings)

# COMMAND ----------

from pyspark.mllib.recommendation import ALS

seed = 5L
iterations = 5
regularizationParameter = 0.1
rank = 4

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE


# COMMAND ----------

seed = 5L
iterations = 10
regularizationParameter = 0.1
rank = 8

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

# COMMAND ----------

seed = 5L
iterations = 15
regularizationParameter = 0.2
rank = 10

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

# COMMAND ----------

seed = 5L
iterations = 20
regularizationParameter = 0.1
rank = 12

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

# COMMAND ----------

seed = 5L
iterations = 25
regularizationParameter = 0.2
rank = 16

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

# COMMAND ----------

seed = 5L
iterations = 16
regularizationParameter = 0.2
rank = 12

myModel = ALS.train(trainingRDD, rank=rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (ISBN, UserID, rating):(ISBN, UserID))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

# COMMAND ----------


