from os import listdir
from os.path import isfile, join
import sys
from datetime import datetime
import xml.etree.ElementTree as ET
import colorsys

def process(xmlfile, src_file, susp_file, src_dir, susp_dir, outdir, color_palette):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    susp_file = root.attrib['reference']
    encoding = 'utf-8'
    with open(f'{src_dir}{src_file}', 'r', encoding=encoding) as f:
        src_text = f.read()
    with open(f'{susp_dir}{susp_file}', 'r', encoding=encoding) as f:
        susp_text = f.read()

    src_split_index = 0
    susp_split_index = 0
    src_html = ""
    susp_html = ""
    for i, detection in enumerate(root):
        # print(detection, detection.attrib)
        if detection.attrib['name'] == 'detected-plagiarism':
            src_file = detection.attrib['source_reference']
            src_length = (int)(detection.attrib['source_length'])
            src_offset = (int)(detection.attrib['source_offset'])
            src_offset_end = src_offset + src_length

            susp_length = (int)(detection.attrib['this_length'])
            susp_offset = (int)(detection.attrib['this_offset'])
            susp_offset_end = susp_offset + susp_length

            src_html += src_text[src_split_index:src_offset] 
            src_html += f'<p class="detection" style="color:{color_palette[i%10]}; font-weight:bold;">'
            src_html += src_text[src_offset:src_offset_end] + f'</p>'

            susp_html += susp_text[susp_split_index:susp_offset] 
            susp_html += f'<p class="detection" style="color:{color_palette[i%10]}; font-weight:bold;">' 
            susp_html += susp_text[susp_offset:susp_offset_end] + f'</p>'

            src_split_index = src_offset_end
            susp_split_index = susp_offset_end
            # print(f'=== DETECTIONS: SRC:[{src_offset}:{src_offset_end}] ---- SUSP:[{susp_offset}:{susp_offset_end}]')
            # print('===== SOURCE TEXT =====')
            # print(src_text[src_offset:src_offset_end])
            # print()
            # print('===== SUSPICIOUS TEXT =====')
            # print(susp_text[susp_offset:susp_offset_end])
            # print("=============================================\n\n")
    src_html += src_text[src_split_index:]
    src_html = src_html.replace('\n', '<br>')
    susp_html += susp_text[susp_split_index:]
    susp_html = susp_html.replace('\n', '<br>')
    
    # print(f'=== DETECTIONS: SRC:[{src_offset}:{src_offset_end}] ---- SUSP:[{susp_offset}:{susp_offset_end}]')
    # print('===== SOURCE TEXT =====')
    # print(src_text[src_offset:src_offset_end])
    # print()
    # print('===== SUSPICIOUS TEXT =====')
    # print(susp_text[susp_offset:susp_offset_end])
    # print("=============================================\n\n")

    out_file = f'{outdir}{susp_file}-{src_file}.xml_{len(root)}.html'.replace('.txt', '')
    full_html = f"""
    <!DOCTYPE html>
    <head>
    <title>{susp_file}-{susp_dir}</title>
    </head>
    <body style="background-color: rgba(0,0,0, .35);">
        <h1>{len(root)} detection(s) found.</h1>
        <div style="width:100%">
            <div style="width: 50%;
                        float: left;
                        display: inline-block;
                        max-height: 700px;
                        overflow-y: scroll;">
                <h1>Source file: {src_file}</h1>
                <p>{src_html}</p>
            </div>
            <div style="width: 50%;
                        float: left;
                        display: inline-block;
                        max-height: 700px;
                        overflow-y: scroll;">
                <h1>Suspicious file: {susp_file}</h1>
                <p>{susp_html}</p>
            </div>
        </div>
    </body>
    </html>
    """
    # print("==== WHAT YOU GOT HERE ===", out_file)
    with open(f'{out_file}', 'w', encoding=encoding) as f:
        f.write(full_html)
        # print(f'{out_file} saved')
    f.close()

# Main
# ====

if __name__ == "__main__":
    color_palette=['#fff100', '#ff8c00', '#e81123', '#ec008c', '#68217a', '#00188f', '#00bcf2', '#00b294', '#009e49', '#bad80a']
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    if len(sys.argv) == 4:
        srcdir = sys.argv[1]
        suspdir = sys.argv[2]
        outdir = sys.argv[3]
        if outdir[-1] != "/" or outdir[-1] != "\\":
            outdir+="/"
        files = [f for f in listdir(outdir) if isfile(join(outdir, f))]
        beginning = datetime.now()
        for i, file in enumerate(files):
            if file.endswith('.xml'):
                print(f'{"{0:0.2f}".format(i / len(files) * 100)}% Processing file {file}... ')
                outfile = file.split('-')
                susp_file = f'{outfile[0]}-{outfile[1]}.txt'
                src_file = f'{outfile[2]}-{outfile[3].split(".")[0]}.txt'            
                # if i > 50:
                    # break
                process(join(outdir, file), src_file, susp_file, srcdir, suspdir, outdir, color_palette=color_palette)

        print(f"=== DONE ! Total times for Task 4 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./task4_paragraph2vec_viewer.py {src-dir} {susp-dir} {out-dir}"]))