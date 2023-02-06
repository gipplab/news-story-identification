import string
import xml.etree.ElementTree as ET

class ECBDocument():
    def __init__(self):
        self.doc_id = 0
        self.doc_name = ''
        self.sentences = []
        self.text = []
        self.sentenceIds = []
        self.whos_human = []
        self.whos_non_human = []
        self.whats = []
        self.whens = []
        self.wheres = []
        self.events = []

    def __str__(self):
        whats = ""
        whos = ""
        whens = ""
        wheres = ""

        for w in self.whats:
            whats += f'\n\t\t\t{w}'
        for w in self.whens:
            whens += f'\n\t\t\t{w}'
        for w in self.wheres:
            wheres += f'\n\t\t\t{w}'
        for w in self.whos_human:
            whos += f'\n\t\t\t{w}'
        for w in self.whos_non_human:
            whos += f'\n\t\t\t{w}'

        str = f"""
        Document #{self.doc_id} from file {self.doc_name}:
        === Content ===\n"""
        for senc in self.sentences:
            str += senc.toStr() + '\n'

        str += f"""
        === The 4 WS (What, When, Where, Who): ===
        {len(self.whats)} whats: {whats}
        {len(self.whos_human) + len(self.whos_non_human)} whos: {whos}
        {len(self.whens)} whens: {whens}
        {len(self.wheres)} wheres: {wheres}
        """

        return str

    def read(self, fileName=None):
        if fileName is None:
            raise Exception(f'File {fileName} is not valid')

        tree = ET.parse(fileName)
        root = tree.getroot()
        self.doc_id = root.attrib['doc_id']
        self.doc_name = root.attrib['doc_name']
        currentSentenceId = 0
        sentenceTokens = []
        sentenceOrderIds = []

        for child in root:
            #   Read tokens
            if child.tag == 'token':
                self.text.append(child.text)
                sentenceId = int(child.attrib['sentence'])
                self.sentenceIds.append(sentenceId)
                if sentenceId != currentSentenceId:
                    newSentence = Sentence(id=currentSentenceId, tokens=sentenceTokens, order=sentenceOrderIds)
                    self.sentences.append(newSentence)
                    sentenceTokens = []
                    sentenceOrderIds = []
                    currentSentenceId = sentenceId
                    # print(f'Sentence read: {newSentence}')
                sentenceTokens.append(child.text)
                sentenceOrderIds.append(child.attrib['number'])

            #   Store 4W
            if child.tag == 'Markables':
                # print(f'Markables:')
                for markable in child:
                    # print(markable.tag, markable.tag in MENTION_ACTIONS, markable.attrib)
                    type = markable.tag
                    content = ""
                    ids = []
                    sentenceIds = []
                    m_id = markable.attrib['m_id']
                    for token in markable:
                        content += self.text[int(token.attrib['t_id']) - 1] + ' '
                        ids.append(token.attrib['t_id'])
                        sentenceIds.append(self.sentenceIds[int(token.attrib['t_id']) - 1])

                    if len(sentenceIds):
                        inMultipleSentence = False
                        id1 = sentenceIds[0]
                        for id in sentenceIds:
                            if id != id1:
                                inMultipleSentence = True
                        sentenceIds = sentenceIds if inMultipleSentence else sentenceIds[0]

                    # WHAT
                    if markable.tag in MENTION_ACTIONS:
                        self.whats.append(Mention(content, type, ids, m_id, sentenceIds))
                    # WHEN
                    if markable.tag in MENTION_TIMES:
                        self.whens.append(Mention(content, type, ids, m_id, sentenceIds))
                    # WHERE
                    if markable.tag in MENTION_LOCATIONS:
                        self.wheres.append(Mention(content, type, ids, m_id, sentenceIds))
                    # WHO-HUMAN
                    if markable.tag in MENTION_HUMANS:
                        self.whos_human.append(Mention(content, type, ids, m_id, sentenceIds))
                    # WHO-NON-HUMAN
                    if markable.tag in MENTION_NON_HUMANS:
                        self.whos_non_human.append(Mention(content, type, ids, m_id, sentenceIds))
                    # print(int(token.attrib['t_id']) + 1, self.text[int(token.attrib['t_id'])])

            if child.tag == 'Relations':
                pass

        #   Added last sentence since the 1st loop will not save it
        newSentence = Sentence(id=currentSentenceId, tokens=sentenceTokens, order=sentenceOrderIds)
        self.sentences.append(newSentence)

        # print(self)
        self.make_events()

    def make_events(self):
        for s in self.sentences:
            id = s.id
            mentions = []
            for m in self.whos_human:
                if m.sentenceId == id:
                    mentions.append(m)

            for m in self.whos_non_human:
                if m.sentenceId == id:
                    mentions.append(m)

            for m in self.whats:
                if m.sentenceId == id:
                    mentions.append(m)

            for m in self.whens:
                if m.sentenceId == id:
                    mentions.append(m)

            for m in self.wheres:
                if m.sentenceId == id:
                    mentions.append(m)

            if len(mentions):
                self.events.append(Event(s, mentions))

    def get_events(self):
        return self.events
        
    def print_events(self):
        print('\t\tPossible events:\n')
        for e in self.events:
            print(e)

class Mention:
    def __init__(self, token=0, type='', t_id=[], m_id=0, sentenceId=0):
        self.type = type
        self.token = token
        self.token_id = t_id
        self.mention_id = m_id
        self.sentenceId = sentenceId

    def __str__(self):
        return f'm_Id: {self.mention_id}'.ljust(10) + f't_ids: {self.token_id}'.ljust(70) + f'TAG: {self.type}'.ljust(40) + f'Texts: {self.token}'.ljust(40) + f'#Sentence: {self.sentenceId}'

    def toStr(self):
        return self.__str__()

class Sentence:
    def __init__(self, id=0, tokens=[], order=[]):
        self.id = id
        self.tokens = tokens
        self.order = order

    def __str__(self):
        str = f"""\t\tSentence #{self.id}:"""
        url_parts = ['http', ':', '/', '//', '-']
        for token in self.tokens:
            if token in string.punctuation or token in url_parts:
                str += f'{token}'
            else:
                str += f' {token}'
        return f'{str}'

    def toStr(self):
        return self.__str__()

class Event:
    def __init__(self, sentence=None, mentions=[]):
        self.sentence = sentence
        self.mentions = mentions

    def __str__(self):
        str = f'{self.sentence.toStr()} + \n'
        for m in self.mentions:
            str += f'\t\t{m.toStr()}\n'

        return str

### THESE TAGS ARE DEFINED IN `Guidelines for ECB+ Annotation of Events and their Coreference`
### http://www.newsreader-project.eu/files/2013/01/NWR-2014-1.pdf

# WHAT
MENTION_ACTIONS = [
    'ACTION_OCCURRENCE',
    'ACTION_PERCEPTION',
    'ACTION_REPORTING',
    'ACTION_ASPECTUAL',
    'ACTION_STATE',
    'ACTION_CAUSATIVE',
    'ACTION_GENERIC',
    'NEG_ACTION_OCCURRENCE',
    'NEG_ACTION_PERCEPTION',
    'NEG_ACTION_REPORTING',
    'NEG_ACTION_ASPECTUAL',
    'NEG_ACTION_STATE',
    'NEG_ACTION_CAUSATIVE',
    'NEG_ACTION_GENERIC'
]

# WHEN
MENTION_TIMES = [
    'TIME_DATE', 
    'TIME_OF_THE_DAY', 
    'TIME_DURATION', 
    'TIME_REPETITION'
]
# WHERE
MENTION_LOCATIONS = [
    'LOC_GEO', 
    'LOC_FAC', 
    'LOC_OTHER',
]
# WHO-HUMAN
MENTION_HUMANS = [
    'HUMAN_PART_PER', 
    'HUMAN_PART_ORG', 
    'HUMAN_PART_GPE', 
    'HUMAN_PART_FAC', 
    'HUMAN_PART_VEH', 
    'HUMAN_PART_MET', 
    'HUMAN_PART_GENERIC'
]
# WHO-NON HUMAN
MENTION_NON_HUMANS = [
    'NON_HUMAN_PART', 'NON_HUMAN_PART_GENERIC'
]

