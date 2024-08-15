from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

def create_spark_session():
    # Create a SparkSession
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("SessionizationExample") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

if __name__ == "__main__":
    spark = create_spark_session()

    # Step 1: Read the input data
    curr_dir = os.getcwd()
    file_path = "Practise/dataset1.txt"  # Adjust the path to match your environment
    df = spark.read.option("header", "true").csv(file_path)
    
    # Show the initial DataFrame
    df.show(truncate=False)

    # Step 2: Convert the Timestamp column to a proper timestamp data type
    df = df.withColumn("timestamp", F.col("Timestamp").cast("timestamp"))

    # Step 3: Define a window specification for partitioning by User_id and ordering by timestamp
    window_spec = Window.partitionBy("User_id").orderBy("timestamp")

    # Step 4: Calculate the time difference between consecutive events for each user
    df = df.withColumn("prev_timestamp", F.lag("timestamp").over(window_spec))
    df = df.withColumn("time_diff", F.col("timestamp").cast("long") - F.col("prev_timestamp").cast("long"))

    # Step 5: Determine when a new session should start
    session_start_condition = (F.col("time_diff").isNull() | 
                               (F.col("time_diff") > 1800) |  # 30 minutes timeout
                               (F.sum("time_diff").over(window_spec) > 7200))  # 2 hours max session

    df = df.withColumn("new_session", F.when(session_start_condition, 1).otherwise(0))

    # Step 6: Create a running session count
    df = df.withColumn("session_number", F.sum("new_session").over(window_spec))

    # Step 7: Generate the session_id
    df = df.withColumn("session_id", F.concat(F.col("User_id"), F.lit("_s"), F.col("session_number")))

    # Intermediate Result: Show the DataFrame after sessionization
    df.select("Timestamp", "User_id", "session_id").show(truncate=False)

    # Step 8: Save the resultant DataFrame to a Parquet file
    df.select("Timestamp", "User_id", "session_id").write.mode("overwrite").parquet("/home/project/output/sessionized_data.parquet")

    # Stop the SparkSession
    spark.stop()
