src_dir = './data/pan14-text-alignment-test-corpus3-2014-05-14/src/'
src_fixed_dir = './data/pan14-text-alignment-test-corpus3-2014-05-14/src_fixed/'
susp_dir = './data/pan14-text-alignment-test-corpus3-2014-05-14/susp/'
susp_fixed_dir = './data/pan14-text-alignment-test-corpus3-2014-05-14/susp_fixed/'

from os import listdir
from os.path import isfile, join

# Fixed source documents
src_files = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]
for f in src_files:
    reader = open(join(src_dir, f), "r", encoding="utf-8")
    text = reader.read()
    reader.close()
    writer = open(src_fixed_dir + f, "w", encoding="utf-8")
    writer.write(text)
    writer.close()

# Fixed suspicious documents
susp_files = [f for f in listdir(susp_dir) if isfile(join(susp_dir, f))]
for f in susp_files:
    reader = open(join(susp_dir, f), "r", encoding="utf-8")
    text = reader.read()
    reader.close()
    writer = open(susp_fixed_dir + f, "w", encoding="utf-8")
    writer.write(text)
    writer.close()
