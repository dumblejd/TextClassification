This application implements Text Classification method by using big data. Which can generate a jar file that can be run on AWS instance.
 
1. How to run my code?
I am running under m5.2xlarge instance

Create a step with type: Spark Application

Sumbit option: 
--class TextRank --master yarn --executor-memory 8g --driver-memory 8g

5 Arguments: 
S3_Path_Of_Dataset
S3_Path_Of_WordsVector
Your_S3_Path_Of_Result  (It should not been created yet)
Ratio_For_Evaluation
Iteration_for_TextRank

The bucket: dibigdata is temporarily public
Example:
s3://dibigdata/articles1_small.csv 
s3://dibigdata/glove.6B.100d.txt 
s3://dibigdata/result_finalcheck 
0.3 
50