import re

# Read the current main.py
with open('app/main.py') as f:
    content = f.read()

# Find and replace the general terms check in the /ask endpoint
old_check = """    if not scored_indices:
        # Check if query is a general high-level query about the paper
        general_terms = {"about", "overview", "summary", "topic", "work", "findings", "method"}
        if any(w in tokenized_query for w in general_terms) and len(chunks) > 0:
            scored_indices = list(range(min(3, len(chunks))))
        else:
            return {"answer": "Not found in this paper.", "sources": []}"""

new_check = """    if not scored_indices:
        # Check if query is a general high-level query about the paper
        general_terms = {"about", "overview", "summary", "topic", "work", "findings", "method"}
        # Check if any general term appears in the query string (handles stemming/variations like "method"/"methodology")
        if any(term in query.lower() for term in general_terms) and len(chunks) > 0:
            scored_indices = list(range(min(3, len(chunks))))
        else:
            return {"answer": "Not found in this paper.", "sources": []}"""

if old_check in content:
    content = content.replace(old_check, new_check)
    with open('app/main.py', 'w') as f:
        f.write(content)
    print("Fix applied successfully")
else:
    print("Old check not found - may already be modified or format differs")