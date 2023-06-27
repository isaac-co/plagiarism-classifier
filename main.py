from textMinator.processing import TextProcessor
from textMinator.similarity import TextSimilarity
import os

# Define working paths 
cwd = os.getcwd()
base_path = os.path.join(cwd, 'docs', 'documentos-plagio')
db_path = os.path.join(cwd, 'docs', 'documentos-genuinos')

# Create lists of paths for files
base_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) 
                 if os.path.isfile(os.path.join(base_path, f))]

db_paths = [os.path.join(db_path, f) for f in os.listdir(db_path) 
                   if os.path.isfile(os.path.join(db_path, f))]

# Instance TextSimilarity
txtSim = TextSimilarity()

# Compare 1x1
txtSim.compare_documents(base_paths[0], db_paths[0], ngrams=[1,2], predict=True)

# Compare 1xN or MxN 
labels = txtSim.compare_document_paths(base_paths[0:5], db_paths, ngrams=[1,2], predict=True, html=True)

print(labels)