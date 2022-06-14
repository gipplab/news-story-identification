# Output format description
Example an output file has a name like:
suspicious-document00001-source-document01256.xml

That means the source document is source-document01256.txt in the source folder specified when training, and the suspicious document is suspicious-document00001.txt in the susp folder specified when training.

Example output content for the file suspicious-document00001-source-document01256.xml:
<?xml version="1.0" encoding="utf-8"?><document reference="suspicious-document00001.txt">
    <feature name="detected-plagiarism" 
        source_length="430" source_offset="1598"        
        source_reference="source-document01256.txt" 
        this_length="430" 
        this_offset="2127"/>
</document>

The source text starts from the position `1598` to `1598+430` (2028) in the source document.
The suspicious text starts from position `2127` to `2127+430` (2557) in the suspicious document.

Content suspicious-document00001.txt: [2127, 2557]
 The severity of the signs depends upon the extent of the tumor and on whether the cancer has caused 
changes in organ function. In many cases, the only noticeable sign is an enlargement of the lymph nodes 
under the neck, behind the knees or in front of the shoulders. 
Other organs, such as the liver, spleen and bone marrow can be involved as well. 
TOP GASTROINTESTINAL: A second form is involvement of the gastrointestinal tract.

Content source-document01256.txt: [1598, 2028]
akness or difficulty breathing. 
The severity of the signs depends upon the extent of the tumor and on whether the cancer has 
caused changes in organ function. In many cases, the only noticeable sign is an enlargement of the 
lymph nodes under the neck, behind the knees or in front of the shoulders. Other organs, 
such as the liver, spleen and bone marrow can be involved as well.
TOP
GASTROINTESTINAL:
A second form is involvement