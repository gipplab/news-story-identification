from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
titleshort = "Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008."

title = "Lindsay Lohan Leaves Betty Ford, Checks Into Malibu Rehab"
lead = "The death toll from a powerful Taliban truck bombing at the German consulate in Afghanistan's Mazar-i-Sharif city rose to at least six Friday, with more than 100 others wounded in a major militant assault."
text = """Lindsay Lohan has left the Betty Ford Center and is moving to a rehab facility in Malibu, Calif., Access Hollywood has confirmed. A spokesperson for The Los Angeles Superior Court confirmed to Access that a judge signed an order yesterday allowing the transfer to Cliffside, where she will continue with her 90- day court- mandated rehab. Lohan ’ s attorney, Shawn Holley, spoke out about the move. “ Lindsay is grateful for the treatment she received at the Betty Ford Center. She has completed her course of treatment there and looks forward to continuing her treatment and building on the foundation established at Betty Ford, ” Holley said in a statement to Access. The actress checked into the Betty Ford Center in May as part of a plea deal stemming from her June 2012 car accident case.
"""
date_publish = 'June 13, 2013 4: 59 PM EDT'

preprocessor = Preprocessor('http://46.101.247.46:9000')
extractor = MasterExtractor(preprocessor=preprocessor)
doc = Document.from_text(text=text, date=date_publish)
doc = extractor.parse(doc)

print(f"Text: {text}")
print(f"Date: {date_publish}")
print("=====Result=====")
#top_who_answer = doc.get_top_answer('who').get_parts_as_text()
#print(f"Who: {top_who_answer}")
#top_what_answer = doc.get_top_answer('what').get_parts_as_text()
#print(f"What: {top_what_answer}")
#top_when_answer = doc.get_top_answer('when').get_parts_as_text()
#print(f"When: {top_when_answer}")
#top_where_answer = doc.get_top_answer('where').get_parts_as_text()
#print(f"Where: {top_where_answer}")
#top_why_answer = doc.get_top_answer('why').get_parts_as_text()
#print(f"Why: {top_why_answer}")
#top_how_answer = doc.get_top_answer('how').get_parts_as_text()
#print(f"How: {top_how_answer}")
whos = doc.get_answers('who')
whats = doc.get_answers('what')
whens = doc.get_answers('when')
wheres = doc.get_answers('where')
#whys = doc.get_answers('why')
print(f"{len(whos)} WHO answers:")
for w in whos:
    print(w.get_parts_as_text())
print(f"{len(whats)} WHAT answers:")
for w in whats:
    print(w.get_parts_as_text())
print(f"{len(whens)} WHEN answers:")
for w in whens:
    print(w.get_parts_as_text())
print(f"{len(wheres)} WHERE answers:")
for w in wheres:
    print(w.get_parts_as_text())
#print(f"{len(whys)} WHY answers")

"""
Text: Lindsay Lohan has left the Betty Ford Center and is moving to a rehab facility in Malibu, Calif., Access Hollywood has confirmed. A spokesperson for The Los Angeles Superior Court confirmed to Access that a judge signed an order yesterday allowing the transfer to Cliffside, where she will continue with her 90- day court- mandated rehab. Lohan ’ s attorney, Shawn Holley, spoke out about the move. “ Lindsay is grateful for the treatment she received at the Betty Ford Center. She has completed her course of treatment there and looks forward to continuing her treatment and building on the foundation established at Betty Ford, ” Holley said in a statement to Access. The actress checked into the Betty Ford Center in May as part of a plea deal stemming from her June 2012 car accident case.

Date: June 13, 2013 4: 59 PM EDT
=====Result=====
3 WHO answers:
Lindsay Lohan
Lindsay
Lohan
4 WHAT answers:
has left the Betty Ford Center and is moving to a rehab facility
is grateful for the treatment
received at the Betty Ford Center
has completed her course and looks forward to continuing her treatment and building
1 WHEN answers:
June 2012
2 WHERE answers:
Betty Ford Center
Cliffside
"""