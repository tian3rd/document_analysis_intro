import os

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity


index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=True)

sim = TFIDF_Similarity
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# TODO run queries
"""
You will need to:
    1. Read in the topics_file.
    2. For each line in the topics file create a query string (note each line has both a query_id and query_text,
       you just want to search for the text)  and run this query on index with index.run_query().
    3. Write the results of the query to runs_file IN TREC_EVAL FORMAT
        - Trec eval format requires that each retrieval is on a separate line of the form
          query_id Q0 document_id rank similarity_score MY_IR_SYSTEM
"""

