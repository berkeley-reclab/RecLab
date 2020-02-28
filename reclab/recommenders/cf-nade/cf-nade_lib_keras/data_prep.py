import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, BlockMatrix


from indexes import create_user_index, create_doc_index, load_indexes, map_recommendations

data_path = '/Users/Guo/Documents/Data/ml-1m/ratings.dat'


def read_data(sc,
	data_file,
	delimiter='::'):
	"""
	Read the data into an RDD of tuple (usrId, productId, rating).
	Args:
		sc: An active SparkContext.
		data_file: A (delimiter) separated file.
		delimiter: The delimiter used to separate the 3 fields of the input file. Default is ','.
	Returns:
		ui_mat_rdd: The UI matrix in an RDD.
	"""

	data = sc.textFile(data_file)
	header = data.first()
	ui_mat_rdd = data.filter(lambda row: row != header) \
		.map(lambda x: (int(x.split(delimiter)[0]),int(x.split(delimiter)[1]),float(x.split(delimiter)[2])))

	return ui_mat_rdd


if __name__ == "__main__":

	sc = SparkContext()
	spark = SparkSession(sc)

	ui_mat_rdd = read_data(sc, data_path, delimiter='::') \
		.sample(False,1.0,seed=0) \
		.persist()

	"""
	print ('Creating usr and doc indexes...')
	user_index = create_user_index(ui_mat_rdd)
	print ('user_index check')
	print (type(user_index))
	print (len(user_index))

	doc_index = create_doc_index(ui_mat_rdd)
	b_uidx = sc.broadcast(user_index)
	b_didx = sc.broadcast(doc_index)
	print(b_didx.value)
	assert False
	for i, (usrId, docId, value) in enumerate(ui_mat_rdd.collect()):
		try:
			val1 = b_uidx.value[usrId]
			val2 = b_didx.value[docId]
		except Exception as e:
			print(e)
			print(i, usrId, docId)
	assert False

	ui_mat_rdd = ui_mat_rdd.map(lambda usrId,docId,value: b_uidx.value[usrId],b_didx.value[docId],value)
	# ui_mat_rdd = ui_mat_rdd.map(lambda value: value) 
	# ui_mat_rdd = ui_mat_rdd.map(lambda usrId: b_uidx.value[usrId])
	# ui_mat_rdd = ui_mat_rdd.map(lambda docId: b_didx.value[docId])
	"""
	num_users = ui_mat_rdd.map(lambda entry: entry[0]) \
		.distinct() \
		.count()
	num_movies = ui_mat_rdd.map(lambda entry: entry[1]) \
		.distinct() \
		.count()
	print ('users:',num_users,'products:',num_movies)


	#Construct the dataframe
	df = spark.createDataFrame(ui_mat_rdd,['userId','movieId','value'])

	ui_mat_rdd.unpersist()


	print ('Splitting data set...')
	df = df.orderBy(F.rand())

	train_df, test_df = df.randomSplit([0.9, 0.1], 
		seed=45)
	train_df, val_df = train_df.randomSplit([0.95, 0.05], 
		seed=45)

	train_df = train_df.withColumn('flag', F.lit(0))
	val_df = val_df.withColumn('flag', F.lit(1))
	val_df = val_df.union(train_df)
	test_df = test_df.withColumn('flag', F.lit(2))
	test_df = test_df.union(train_df)
	test_df = test_df.union(val_df)

	train_size = train_df.count()
	val_size = val_df.count()
	test_size =  test_df.count()

	train_df.show()
	print (train_size,'training examples')
	print (val_size,'validation examples')
	print (test_size,'testing example')




	train_examples = train_df.select("movieId", F.struct(["userId","value","flag"]).alias("ranking")) \
		.groupby('movieId') \
		.agg(F.collect_list('ranking').alias('rankings'))
	val_examples = val_df.select("movieId", F.struct(["userId","value","flag"]).alias("ranking")) \
		.groupby('movieId') \
		.agg(F.collect_list('ranking').alias('rankings'))
	test_examples = test_df.select("movieId", F.struct(["userId","value","flag"]).alias("ranking")) \
		.groupby('movieId') \
		.agg(F.collect_list('ranking').alias('rankings'))

	train_examples.show()
	val_examples.show()
	test_examples.show()


	train_examples.coalesce(1).write.json(path="/Users/Guo/Documents/Data/ml-1m/train_set",
		mode='overwrite')
	val_examples.coalesce(1).write.json(path="/Users/Guo/Documents/Data/ml-1m/val_set",
		mode='overwrite')
	test_examples.coalesce(1).write.json(path="/Users/Guo/Documents/Data/ml-1m/test_set",
		mode='overwrite')
