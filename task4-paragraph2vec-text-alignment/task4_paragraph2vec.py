#!/usr/bin/env python
""" Plagiarism detection for near-duplicate plagiarism.

    This program provides the baseline for the PAN 2013 Plagiarism Detection
    task and can be used as outline for a plagiarism detection program.
"""
__author__ = 'Arnd Oberlaender'
__email__ = 'arnd.oberlaender at uni-weimar dot de'
__version__ = '1.1'

import os
import sys
from datetime import datetime
from paragraph2vec import Paragraph2Vec

# Helper functions
# ================

""" The following functions are some simple helper functions you can utilize
and modify to fit your own program.
"""


# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    if len(sys.argv) == 5:
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir+="/"
        lines = open(sys.argv[1], 'r').readlines()
        beginning = datetime.now()
        threshold = 0.99
        vector_size = 20
        alpha = 0.025
        for i, line in enumerate(lines):
            start_time = datetime.now()
            susp, src = line.split()
            # if(src == 'source-document02004.txt' and susp == 'suspicious-document00001.txt'):
            print(f'{"{0:0.2f}".format(i / len(lines) * 100)}% Processing pair {src}-{susp}... ', end='')

            model = Paragraph2Vec(                
                os.path.join(suspdir, susp),
                os.path.join(srcdir, src), 
                outdir,
                threshold=threshold,
                alpha=alpha,
                vector_size=vector_size,
                debug=False   
            )
            model.process()
            print(f'done! {datetime.now() - start_time} elapsed')

            # if i >= 10:
            #     break
        print(f'Model configuration: \n\tThreshold: {threshold}\n\tVector-size: {vector_size}\n\tAlpha: {alpha}')
            
        print(f"=== DONE ! Total times for Task 4 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir}"]))