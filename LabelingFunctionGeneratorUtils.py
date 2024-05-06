from snorkel.labeling import labeling_function
from textblob import TextBlob
from thefuzz import fuzz
import re

'''

### Types of Labeling Functions and their respective options

1. Sentiment of a document/transcript
2. Presence or absence of a named entity
3. Topic Search: Topics are curated from the output of BERTopic.
4. Keyword Matching (Positive LF)
5. Keyword Not Matching (negative LF)
6. Gazetteer


### Operations performed by each function:

1. Check for presence of a SpaCy Named Entity Tag
2. Check for the presence of a keyword or a set of keyword
3. Check against a long list of names/keywords (Gazatteers)
4. Check if sentiment is within a specific threshold



### Parameters required to define an LF

- Name of the LF
- Label Name
- Type of the LF
  - Keyword Matching (Positive LF)
    - FunctionName
    - Matching Type (strict, fuzzy)
    - List of words to match
    - Return Type (Label)
  - Keyword Not Matching (Negative LF)
    - FunctionName
    - Matching Type (strict, fuzzy)
    - List of words to match
    - Return Type (Label)
  - NER Tag Presence
    - FunctionName
    - NER Tag
    - Return Type (Label)
  - NER Tag Absence
    - FunctionName
    - NER Tag
    - Return Type (Label)
  - Sentiment Check
    - FunctionName
    - TextBlob: 
      - `subjectivity`->[0.0, 1.0] where 0 is purely objective and 1 is purely subjective
      - `polarity`->[-1.0, +1.0] where -1 is negative and +1 is positive
      
'''


def NERTagPresence(**kwargs):
    '''
    Detects if a specific SpaCy NER tagged token is present in the transcript
    and returns LABEL
    if no tag : ABSTAIN
    
    Params:
    LabelingFunctionName:str
    NERTag:str
    ReturnLabel:int (class label)
    '''

    LabelingFunctionName = kwargs['LabelingFunctionName']
    NERTag = kwargs['NERTag']
    ReturnLabel = kwargs['ReturnLabel']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def CheckNERTagPresence(InputSpacyDoc):
        for each_ent in InputSpacyDoc.ents:
            if each_ent.label_ == NERTag:
                return ReturnLabel
        return ABSTAIN
    
    CheckNERTagPresence.__name__ = LabelingFunctionName
    
    return CheckNERTagPresence
    
def NERTagAbsence(**kwargs):
    '''
    Detects if a specific SpaCy NER tagged token is absent in the transcript
    and returns LABEL if the tag is absent
    if tag is present, then: ABSTAIN
    
    Params:
    LabelingFunctionName:str
    NERTag:str
    ReturnLabel:int (class label)
    '''

    LabelingFunctionName = kwargs['LabelingFunctionName']
    NERTag = kwargs['NERTag']
    ReturnLabel = kwargs['ReturnLabel']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def CheckNERTagAbsence(InputSpacyDoc):
        for each_ent in InputSpacyDoc.ents:
            if each_ent.label_ == NERTag:
                return ABSTAIN
        return ReturnLabel
    
    CheckNERTagAbsence.__name__ = LabelingFunctionName
    
    return CheckNERTagAbsence


def SentimentChecker(**kwargs):
    
    '''
    Detects if a document has a sentiment score (polarity) within
    the specified threshold (not including the specified value)
    
    Returns the label if the sentiment value extracted from
    TextBlob matches the sentiment threshold
    
    Else ABSTAIN
    
    subjectivity->[0.0, 1.0] where 0 is purely objective and 1 is purely subjective
    
    SubjectivityLower: (float) -> lower threshold for subjectivity
    SubjectivityUpper: (float) -> upper threshold for subjectivity
    
    polarity->[-1.0, +1.0] where -1 is negative and +1 is positive

    PolarityLower: (float) -> lower threshold for polarity
    PolarityUpper: (float) -> upper threshold for polarity
    
    LabelingFunctionName:str
    ReturnLabel:int (class label)
    
    '''
    
    
    LabelingFunctionName = kwargs['LabelingFunctionName']
    
    #subjectivity->[0.0, 1.0] where 0 is purely objective and 1 is purely subjective
    SubjectivityLower = kwargs['SubjectivityLower']
    SubjectivityUpper = kwargs['SubjectivityUpper']
    
    #polarity->[-1.0, +1.0] where -1 is negative and +1 is positive
    PolarityLower = kwargs['PolarityLower']
    PolarityUpper = kwargs['PolarityUpper']
    ReturnLabel = kwargs['ReturnLabel']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def TextBlobSentimentChecker(InputSpacyDoc):
        scores = TextBlob(InputSpacyDoc.text)
        polarity = scores.sentiment.polarity
        subjectivity = scores.sentiment.subjectivity
        if (polarity < PolarityUpper) \
        and (polarity > PolarityLower) \
        and (subjectivity < SubjectivityUpper) \
        and (subjectivity > SubjectivityLower):
            return ReturnLabel 
        else:
            return ABSTAIN
    TextBlobSentimentChecker.__name__ = LabelingFunctionName
    
    return TextBlobSentimentChecker




def KeywordPresenceExactMatch(**kwargs):
    '''
    Detects if a keyword from a list of topics/keywords 
    is present in the text document
    and returns LABEL if present
    if not present : ABSTAIN
    
    Params:
    LabelingFunctionName:str
    ListOfKeywords:[str]
    ReturnLabel:int (class label)
    
    '''
    

    LabelingFunctionName = kwargs['LabelingFunctionName']
    ListOfKeywords = kwargs['ListOfKeywords']
    ReturnLabel = kwargs['ReturnLabel']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def CheckForKeywordExactMatch(InputSpacyDoc):
        for each_keyword in ListOfKeywords:
            if re.search('([^a-z0-9])' + each_keyword + '([^a-z0-9])', InputSpacyDoc.text.lower()) :
                return ReturnLabel
        return ABSTAIN

    CheckForKeywordExactMatch.__name__ = LabelingFunctionName
    return CheckForKeywordExactMatch
    
    
def NamedEntityFuzzyMatch(**kwargs):
    '''
    Detects if a Named Entity from a list of Gazetteer 
    is present in the text document using fuzzy search
    and returns LABEL if present
    if not present : ABSTAIN
    
    Params:
    LabelingFunctionName:str
    ListOfKeywords:[str]
    ReturnLabel:int (class label)
    NERTag:str
    FuzzyMatchThreshold: int (value between 0 to 100)
    
    '''
    
    
    LabelingFunctionName = kwargs['LabelingFunctionName']
    ListOfKeywords = kwargs['ListOfKeywords']
    ReturnLabel = kwargs['ReturnLabel']
    FuzzyMatchThreshold = kwargs['FuzzyMatchThreshold']
    NERTag = kwargs['NERTag']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def CheckForNamedEntityFuzzyMatch(InputSpacyDoc):
        NamedEntityInDoc = ''
        max_ratio = 0
        for each_ent in InputSpacyDoc.ents:
            if each_ent.label_ == NERTag:
                NamedEntityInDoc = each_ent.text

                for each_name in list(set(ListOfKeywords)):
                    fuxzzRatio = fuzz.token_sort_ratio(str(NamedEntityInDoc).lower(), each_name.lower())
                    if  fuxzzRatio > max_ratio:
                        max_ratio = fuxzzRatio
                        closest_name = each_name
                if max_ratio >= FuzzyMatchThreshold:
                    #print(closest_name, max_ratio)
                    return ReturnLabel
        return ABSTAIN

    CheckForNamedEntityFuzzyMatch.__name__ = LabelingFunctionName
    return CheckForNamedEntityFuzzyMatch

    
def KeywordAbsenceExactMatch(**kwargs):
    '''
    Detects if a keyword from a list of topics/keywords 
    is absent in the text document
    and returns LABEL if absent
    if present : ABSTAIN
    
    Params:
    LabelingFunctionName:str
    ListOfKeywords:[str]
    ReturnLabel:int (class label)
    
    '''
    
    
    LabelingFunctionName = kwargs['LabelingFunctionName']
    ListOfKeywords = kwargs['ListOfKeywords']
    ReturnLabel = kwargs['ReturnLabel']
    ABSTAIN = -1
    @labeling_function(name = LabelingFunctionName)
    def CheckForKeywordAbsenceExactMatch(InputSpacyDoc):
        for each_keyword in ListOfKeywords:
            if re.search('([^a-z0-9])' + each_keyword + '([^a-z0-9])', InputSpacyDoc.text.lower()) :
                return ABSTAIN
        return ReturnLabel

    CheckForKeywordAbsenceExactMatch.__name__ = LabelingFunctionName
    return CheckForKeywordAbsenceExactMatch