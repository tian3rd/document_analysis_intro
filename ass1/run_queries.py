import os

from nltk.util import pr

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity

import re

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=True)

sim = TFIDF_Similarity
# sim = TF_Similarity
index.set_similarity(sim)
# sim().set_modes("n", "n", "c")
# index.similarity_measure.set_modes("n", "n", "c")
print(index.similarity_measure.TF_mode, index.similarity_measure.Norm_mode)
print(f'Setting similarity to {sim.__name__}')

print('Index ready.')

topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.txt')
# runs_file = os.path.join('gov', 'retrieved.txt')


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
# runs_folder = os.path.join('runs')
if not os.path.exists('runs'):
  os.mkdir('runs')
with open(runs_file, 'w') as written, open(topics_file) as f:
    contents = f.readlines()
    print("writing")
    # print(contents)
    for line in contents:
      line = line.split(" ")
      # extract query id like 1, 46
      query_id = line[0]
      # print(query_id)
      line = " ".join(line[1:])
      # get rid of '/' ','
      line = re.findall('\w+', line)
      # get rid of 's' or single char
      line = " ".join(filter(lambda x: len(x) > 1, line))
      # print(line)
      results = index.run_query(line)
      for rank in range(len(results)):
        trecline = "{query_id} Q0 {document_id} {rank} {similarity_score} MY_IR_SYSTEM\n".format(query_id=query_id, document_id=results[rank][0], rank=rank, similarity_score=results[rank][1])
        # print("{rank} written!".format(rank=rank))
        written.write(trecline)

# print("tokens: {0}; docs: {1}".format(len(index.postings.token_to_doc_counts), len(index.postings.doc_to_token_counts)))
# for tok in index.postings.token_to_doc_counts:
#   print("token example: {0}, doc example: {1}".format(tok, index.postings.token_to_doc_counts[tok]))
#   break
# for doc in index.postings.doc_to_token_counts:
#   print("doc example: {0}, token example: {1}".format(doc, index.postings.doc_to_token_counts[doc]))
#   break