from modules.text_alignment.evaluate import evaluate
from modules.text_alignment.paragraph2vec import Paragraph2VecWrapper
from modules.text_alignment.viewer import generate_viewer
from modules.text_alignment.sbert import SBertWrapper
from config import Config

class TextAlignmentProcessor():
    def __init__(self, config: Config):
        self.susp_dir = config['TASK-4']['SuspFolder']
        self.src_dir = config['TASK-4']['SrcFolder']
        self.output_dir = config['TASK-4']['OutputFolder']
        self.alpha = config['TASK-4']['Alpha']
        self.vector_size = config['TASK-4']['VectorSize']
        self.model_name = config['TASK-4']['PretrainedBERT']
        self.method_name = config['TASK-4']['MethodType']
        self.real_truth_dir = config['TASK-4']['TrainingTruthFolder']
        self.pairs = config['TASK-4']['PairPath']
        self.threshold = float(config['TASK-4']['Threshold'])
        self.threshold_length = float(config['TASK-4']['Threshold_Length'])
        self.verbose = config['GLOBAL']['Verbose']

    def start_alignment(self):
        if self.method_name == 'SBert':
            SBertWrapper(
                    pairs=self.pairs,
                    srcdir=self.src_dir,
                    suspdir=self.susp_dir,
                    outdir=self.output_dir,
                    threshold=self.threshold,
                    threshold_length=self.threshold_length,
                    model_name=self.model_name,
                    verbose=self.verbose).process()
        if self.method_name == 'Parahraph2Vec':
            Paragraph2VecWrapper(             
                    pairs=self.pairs,
                    srcdir=self.src_dir,
                    suspdir=self.susp_dir,
                    outdir=self.output_dir,
                    threshold=self.threshold,
                    threshold_length=self.threshold_length,
                    alpha=self.alpha,
                    vector_size=self.vector_size,
                    verbose=self.verbose   
                ).process()
        if self.method_name == 'Sachezperez':   # Winner of the contest PAN2014
            pass
        if self.method_name == 'baseline':      
            pass
        
        self.postprocess()

        if len(self.real_truth_dir) > 0:
            self.evaluate()

    def postprocess(self):
        """ Generate viewers & Evaluate """
        generate_viewer(srcdir=self.src_dir, suspdir=self.susp_dir, outdir=self.output_dir, pairs=self.pairs)

    def evaluate(self):    
        """ Evaluate """
        evaluate(outdir=self.output_dir, true_value_dir=self.real_truth_dir)