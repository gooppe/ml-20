import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext


object WordCounter {
  def main(args: Array[String]) {
    val sc = new SparkContext()
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val textFile = sc.textFile(args(0))
    val counts = textFile.flatMap(line => "\\W+".r.split(line))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .map(item => item.swap)
      .sortByKey(false, 1)
      .map(item => item.swap)
      
    val df = counts.toDF("word", "count")
    df.write.format("csv").save(args(1))
  }
}