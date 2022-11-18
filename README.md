# Data - FunCom

Download data from http://leclair.tech/data/funcom/index_v5.html

The filtered version and the processed version are needed, because the processed
summaries are used by the preprocessing script. Alternatively the comments.json
file from the filtered version could be processed into a tsv file containing the
ids and only the summary of the JavaDoc string.

with following link: 
https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_filtered.tar.gz

## Preprocessing

The file preprocess.py holds the operations used to process the methods of the
FunCom dataset. The actions to process the methods include:

- Comment removal
- Splitting camel case words
- Encasing special characters with blanks (e.g. +, -, ==, ||, ...)
- Trimming whitespaces
- Generation of an AST (for the model Rencos https://github.com/zhangj111/rencos)
