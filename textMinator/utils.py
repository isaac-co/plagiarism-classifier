import iso639
import langid
import re
import os

def get_language(text):
    """Returns the english name of the language
    in which the text is written"""
    # Detect the language
    iso = langid.classify(text)[0]
    # Convert ISO-639 short form to english language name
    language = iso639.to_name(iso)
    # Get only first word in lowercase (for cases like Spanish; Castillian)
    language = re.match(r'^\w*', language)[0].lower()
    
    return language

def classify_by_language(paths):
    """Returns a dictionary with the keys being languages and the values
    a list of documents in that language"""
    language_dict = {}
    
    if isinstance(paths, str) and os.path.isfile(paths):
        with open(paths, encoding="utf8") as f:
            file = f.read()
        language_dict[get_language(file)] = [paths]
        return language_dict
    
    else:
        # Iterate over the document paths
        for file in paths:
            with open(file, encoding="utf8") as f:
                og = f.read()
                language = get_language(og)

                # Check if the language is already a key in the dictionary
                if language in language_dict:
                    language_dict[language].append(file)
                else:
                    language_dict[language] = [file]
                
    return language_dict