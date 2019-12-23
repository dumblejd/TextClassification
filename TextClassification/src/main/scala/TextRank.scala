import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object TextRank {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Text Classification")
      .getOrCreate()
    import spark.sqlContext.implicits._

    val dataFilepath = args(0)
    val modelFilepath = args(1)
    val ratio = args(3).toDouble
    val iter = args(4).toInt
    val temp_dataDF = spark.read.option("header", "true").option("inferSchema", "true").csv(dataFilepath)
    val dataDF = temp_dataDF.filter((($"content" =!= "") || ($"content" =!= null))&&(($"title" =!= "") || ($"title" =!= null)))
    // Extract word vectors
    val wordVectors = spark.sparkContext.textFile(modelFilepath) //.map(_.toLowerCase) ??? why
    val wordVectorDict = wordVectors.map(x => (x.split(" ", 2)(0), x.split(" ", 2)(1))).map(x => (x._1, x._2.split(" ")))
    // remove stop words function
    val stopwords = StopWordsRemover.loadDefaultStopWords("english").toSet

    def remove_stopwords(sentences: Array[String]): Array[String] = {
      var result = Array[String]()
      for (sentence <- sentences) {
        var sentence_new = ""
        for (i <- sentence.split(" ")) {
          if (!stopwords.contains(i)) {
            sentence_new = sentence_new + i + " "
          }
        }
        result = result :+ sentence_new
      }
      result
    }



    var tokenizer = new RegexTokenizer().setInputCol("content").setOutputCol("result_sentences").setPattern("[^.!?]*[^.!?]").setGaps(false).setToLowercase(false)
    val tokenizerDFF = tokenizer.transform(dataDF)
    // display(tokenizerDFF)
    var tokenizer2 = new RegexTokenizer().setInputCol("content").setOutputCol("sentences").setPattern("[^.!?]*[^.!?]").setGaps(false)
    val tokenizerDF = tokenizer2.transform(tokenizerDFF)
    //tokenizerDF.withColumn("sentences",stringify($"sentences")).withColumn("result_sentences",stringify($"result_sentences")).write.format("com.databricks.spark.csv").option("header", "true").csv(args(2)+"/accuracy")

    val cleanRdd = tokenizerDF.select("content","sentences","result_sentences","title").rdd.map(x=>{
      val text = x.get(0).toString
      //   val sentences = remove_stopwords(x.get(1).toList)
      val sentences = remove_stopwords(x.getSeq[String](1).toArray)
      val result_sentences=x.getSeq[String](2).toArray
      var title=x.get(3).toString
      (text,sentences,result_sentences,title)
    })

    //cleanRdd.toDF("content","sentences","result_sentences","title").withColumn("sentences",stringify($"sentences")).withColumn("result_sentences",stringify($"result_sentences")).write.format("com.databricks.spark.csv").option("header", "true").csv(args(2)+"/accuracy")
    val wvd = wordVectorDict.map(x => (x._1, x._2.map(y => y.toDouble)))
    val vertexMap = wvd.collectAsMap()

    //Extract word vectors  100Dimension only
    //Extract word vectors  100Dimension only
    def getVector(sentence:String)={
      var senVec = new Array[Double](100)
      for(i<-sentence.split(" ")){
        if(vertexMap.contains(i)){
          senVec = senVec.zip(vertexMap(i)).map{case(x,y)=>x+y}
        }
      }
      senVec.map(x=>x/(sentence.split(" ").length.toDouble))
      senVec
    }
    def getVectors(sentences: Array[String])={
      var result = Array[Array[Double]]()

      for(sentence<-sentences){
        result= result:+getVector(sentence)
      }
      result
    }

    val rddWithVector = cleanRdd.map(x=>(x._4,x._3,x._1,getVectors(x._2)))
    val dfWithVector = rddWithVector.toDF("title","result_sentences","text","vector")


    //simularity
    def magnitude(x: Array[Double]): Double = {
      math.sqrt(x map (i => i * i) sum)
    }

    def dotProduct(x: Array[Double], y: Array[Double]): Double = {
      (for ((a, b) <- x zip y) yield a * b) sum
    }

    def cosineSimilarity(x: Array[Double], y: Array[Double]): Double = {
      require(x.size == y.size)
      val magx = magnitude(x)
      val magy = magnitude(y)
      if (magx.isNaN || magy.isNaN || magx == 0.0 || magy == 0.0) {
        val min = -1
        min
      }
      else {
        dotProduct(x, y) / (magx * magy)
      }
    }


    def sortByRank(arr1: Array[Double], arr2: Array[Double]) = {
      arr1(1) > arr2(1)
    }

    def get_rank(vectors:Array[Array[Double]])={
      var edges = Array[Array[Double]]()
      //   var :Map[Char,Int] = Map()
      val vertexNum = vectors.length
      val edgesNum = vertexNum*vertexNum
      val alpha = 0.15
      for(start<-0 to vectors.length-1){
        var edge = Array[Double]()
        for(end<-0 to vectors.length-1){
          val cos_similarity = cosineSimilarity(vectors(start),vectors(end))
          edge=edge:+cos_similarity
        }
        edges=edges:+edge
      }
      edges
      var vertexRanks = Array[Double]()
      for(i<-0 to vertexNum-1){
        vertexRanks=vertexRanks:+10.0
      }
      var result = Array[Array[Double]]()
      for(i<-0 to iter){
        for(start<-0 to vectors.length-1){
          var temp = 0.0
          for(end<-0 to vectors.length-1){
            temp=temp+(vertexRanks(end)*edges(start)(end))
          }
          temp=temp/vertexNum*alpha+(1-alpha)/edgesNum.toDouble
          vertexRanks(start)=temp
          //     result=result:+Array(start,temp)
        }
      }
      for(i<-0 to vertexNum-1){
        result=result:+Array(i,vertexRanks(i))
      }
      var res =result.sortWith(_(1)>_(1))
      res
    }


    //extract output
    val resWithRank = rddWithVector.map(x=>(x._1,x._3,x._2,get_rank(x._4),x._4))

    def get_sentences(sentences: Array[String], rank: Array[Array[Double]]) = {
      var res = Array[String]()
      for (i <- 0 to rank.length - 1) {
        res = res :+ sentences(rank(i)(0).toInt)
      }
      res
    }

    val title_toVectorRDD = resWithRank.map(x=>(getVector(x._1),x._2,x._3,x._4,x._5))

    def evaluate_rank(title_vector:Array[Double],vectors:Array[Array[Double]])={
      var cos_similarities = Array[Array[Double]]()
      for(i<-0 to vectors.length-1){
        val cos_similarity = cosineSimilarity(title_vector,vectors(i))
        cos_similarities=cos_similarities:+Array(i,cos_similarity)
      }
      var res = cos_similarities.sortWith(_(1)>_(1))
      res
    }

    val rankWithEvaluation = title_toVectorRDD.map(x=>(x._2,x._3,x._4,evaluate_rank(x._1,x._5)))

    def prediction_similarity(ratio:Double,rank:Array[Array[Double]], evaluate_rank:Array[Array[Double]])={
      //val size = math.min(rank.length,n)
      var n = (rank.length*ratio).toInt
      if(n<1){n=1}
      var res = 0;
      for(i<-0 to n-1){
        for(j<-0 to n-1){
          if(rank(i)(0)==evaluate_rank(j)(0)){
            res=res+1
          }
        }
      }
      val result=res.toDouble/n.toDouble
        result
    }

    //accuracy
    val accuracy=rankWithEvaluation.map(x=>(x._1,get_sentences(x._2,x._3),prediction_similarity(ratio,x._3,x._4)))
    val outerResult = rankWithEvaluation.map(x=>(x._1,get_sentences(x._2,x._3),x._3))
    val d = accuracy.map(x=>(1,x._3)).reduceByKey(_+_).collect()

    val overall_accuracy = d(0)._2/dataDF.count()

    val stringify = udf((vs: Seq[String]) => vs match {
      case null => null
      case _    => s"""[${vs.mkString(",")}]"""
    })

    //res.saveAsTextFile(args(2))
    val resDF = outerResult.map(x=>(x._1,x._2,x._3.flatten.mkString(",")))//.toDF("text","sentences","rank","evaluate_rank")//.withColumn("sentences",stringify($"sentences")).withColumn("rank",stringify($"rank")).withColumn("evaluate_rank",stringify($"evaluate_rank"))
   //outofuse1//val rrr = resDF.select($"text",flatten($"rank"),flatten($"evaluate_rank"))//.withColumn("rank",stringify($"rank")).withColumn("evaluate_rank",stringify($"evaluate_rank"))
   //version2//val rrr =  resDF.toDF("text","rank","evaluate_rank").drop("evaluate_rank").withColumn("rank",stringify($"rank"))
   val rrr =  resDF.toDF("text","rank","vector_rank").withColumn("rank",stringify($"rank"))

    //resDF.select($"text",$"rank",$"evaluate_rank")//.withColumn("rank",stringify($"rank")).withColumn("evaluate_rank",stringify($"evaluate_rank"))

    rrr.write.format("com.databricks.spark.csv").option("header", "true").csv(args(2))
    val accuracyDF = accuracy.toDF("text","sentences","prediction").withColumn("sentences",stringify($"sentences"))
    accuracyDF.write.format("com.databricks.spark.csv").option("header", "true").csv(args(2)+"/accuracy")
    spark.sparkContext.parallelize(Seq(overall_accuracy)).coalesce(1,true).saveAsTextFile(args(2)+"/overallAccuracy"+iter+"_"+ratio+"_"+overall_accuracy)

  }
}
