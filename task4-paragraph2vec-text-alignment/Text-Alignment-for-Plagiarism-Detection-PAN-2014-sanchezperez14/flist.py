'''
Created on Sep 4, 2013

__authors__ = 'Miguel Angel Sanchez Perez, Alexander Gelbukh and Grigori Sidorov'
__email__ = 'masp1988 at hotmail dot com'

The wordlists used in this file:
    sw50_list: see paper: Efstathios Stamatatos. Plagiarism Detection Using Stopword n-grams. Journal of the American Society for Information Science and Technology, Volume 62, Issue 12, December 2011, pages 2512-2527.
    sw_list: from NLTK, www.nltk.org; method stopwords.
'''
class flist:
    def __init__(self):
        self.sw50_list=['the','of','and','a','in','to','is','was','it','for','with','he','be','on','i','that','by','at','you','\'s','are','not','his','this','from','but','had','which','she','they','or','an','were','we','their','been','has','have','will','would','her','n\'t','there','can','all','as','if','who','what','said']
        self.sw_list=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', '\'s', 'n\'t', 'can', 'will', 'just', 'don', 'should', 'now']
    
    def words50(self):
        return self.sw50_list
    
    def words(self):
        return self.sw_list