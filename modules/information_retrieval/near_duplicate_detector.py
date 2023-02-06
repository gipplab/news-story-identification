from sentence_transformers import SentenceTransformer, util
import nltk

class NearDuplicateEvaluator():

    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.90, threshold_length=10, verbose=False):
        sbert_model_list = [
            'all-mpnet-base-v2',                      
            'multi-qa-mpnet-base-dot-v1',             
            'all-distilroberta-v1',                   
            'all-MiniLM-L12-v2',                      
            'multi-qa-distilbert-cos-v1',             
            'all-MiniLM-L6-v2',                       
            'multi-qa-MiniLM-L6-cos-v1',              
            'paraphrase-multilingual-mpnet-base-v2',  
            'paraphrase-albert-small-v2',             
            'paraphrase-multilingual-MiniLM-L12-v2',  
            'paraphrase-MiniLM-L3-v2',                
            'distiluse-base-multilingual-cased-v1',   
            'distiluse-base-multilingual-cased-v2'    
        ]
        self.threshold = threshold
        self.threshold_length = threshold_length
        self.model_name = model_name
        self.verbose = verbose
        # if model_name in sbert_model_list:
        self.model = SentenceTransformer(self.model_name)
        
    def evaluate(self, doc_1: str, doc_2: str) -> bool:
        sentences_1, embeddings_1 = self.preprocess(doc=doc_1)
        sentences_2, embeddings_2 = self.preprocess(doc=doc_2)

        detections = self.detect(doc_1, sentences_1, embeddings_1,
                                doc_2, sentences_2, embeddings_2) 
        return bool(len(detections))

    def cosine(self, doc_1: str, doc_2: str):
        _, embeddings_1 = self.preprocess(doc=doc_1)
        _, embeddings_2 = self.preprocess(doc=doc_2)

        return util.cos_sim(embeddings_1, embeddings_2)

    def preprocess(self, doc: str):
        sentences = self.split_sentences(doc)
        embeddings = self.model.encode(sentences)
        return sentences, embeddings

    def detect(self, src_text, src_sentences, src_embeddings, susp_text, susp_sentences, susp_embeddings):
        pairs = []
        for i, src_emb in enumerate(src_embeddings):
            for j, susp_emb in enumerate(susp_embeddings):
                cos_sim = util.cos_sim(src_emb, susp_emb)
                if cos_sim >= self.threshold:
                    pairs.append((i, j))

        detections = []
        srcIdx = 0
        suspIdx = 0
        if len(pairs) > 0:
            srcS = src_sentences[pairs[0][0]]
            srcIdx = src_text.find(srcS)
            suspS = susp_sentences[pairs[0][1]]
            suspIdx = susp_text.find(suspS)
            new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                  
            for i in range(len(pairs)):
                if i == len(pairs) - 1:  
                    if self.verbose:
                        print(f'[DETECTION-end] Pairs: {pairs[i]}')
                        print(f'[DETECTION-end] Pairs-srcS: {src_sentences[pairs[i][0]]}')
                        print(f'[DETECTION-end] Pairs-suspS: {susp_sentences[pairs[i][1]]}')
                        print(f'[DETECTION-end] {new_detection}:')      
                        print(f'[DETECTION-end] Source:\n {src_text[new_detection[0][0]:new_detection[0][1]]}')
                        print(f'[DETECTION-end] Suspicious:\n {susp_text[new_detection[1][0]:new_detection[1][1]]}')
                    if self.validate_detection(new_detection):
                        detections.append(new_detection)                    
                else:
                    if pairs[i + 1][0] == pairs[i][0] + 1:
                        srcS = src_sentences[pairs[i + 1][0]]
                        srcIdx = src_text.find(srcS)
                        suspS = susp_sentences[pairs[i + 1][1]]
                        suspIdx = susp_text.find(suspS)
                        new_detection[0][1] = srcIdx + len(srcS)
                        new_detection[1][1] = suspIdx + len(suspS)                         
                    else:
                        if self.verbose:
                            print(f'[DETECTION] {new_detection}:')      
                            print(f'[DETECTION] Source:\n {src_text[new_detection[0][0]:new_detection[0][1]]}')
                            print(f'[DETECTION] Suspicious:\n {susp_text[new_detection[1][0]:new_detection[1][1]]}')
                        if self.validate_detection(new_detection):
                            detections.append(new_detection)
                        srcS = src_sentences[pairs[i + 1][0]]
                        srcIdx = src_text.find(srcS)
                        suspS = susp_sentences[pairs[i + 1][1]]
                        suspIdx = susp_text.find(suspS)
                        new_detection = [[srcIdx, srcIdx + len(srcS)], [suspIdx, suspIdx + len(suspS)]]                            
        return detections    

    ##########################################################
    #                                                        #
    #  Source code from https://stackoverflow.com/a/4576110  #
    #                                                        #
    ##########################################################
    def split_sentences(self, input_text=""):
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(input_text)        
        return sentences

    def validate_detection(self, detection):
        return (detection[0][1] > detection[0][0]) and (detection[1][1] > detection[1][0])