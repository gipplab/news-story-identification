cd D:\DevApps\Python379\Lib\site-packages\Giveme5W1H\runtime-resources\stanford-corenlp-full-2017-06-09
java -Xmx5G -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 600000 -threads 5 -maxCharLength 100000 -quiet False -preload tokenize,ssplit,pos,lemma,ner,parse,depparse,mention,coref

REM #java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 600000