#%%
import pandas as pd
import spacy
from collections import defaultdict
import json

# Load the English language model
nlp = spacy.load("en_core_web_sm")

df = pd.read_json("../vast3/public/data.json")

# Function to extract named entities and nouns from the "message" column
def extract_entities(text):
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ('PERSON','ORG','GPE')]
    return entities

def extract_nouns(text):
    
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() not in entities_name]
    return [(noun, 'NOUN') for noun in nouns]


# Extract named entities and nouns from the "message" column
df['Text'] = df['Text'].astype(str)
df['entities'] = df['Text'].apply(extract_entities)

#%%
global entities_name
entities_name = []
for ents in df['entities']:
    if ents:
        for i in ents:
            for j in i[0].split(" "):
                entities_name.append(j.lower())

nouns = df['Text'].apply(extract_nouns)

df['entities'] = df['entities'] + nouns

def format_to_key_value(row):
    key_value_pairs = []
    import ast
    row = ast.literal_eval(row)
    for item in row:
        if isinstance(item, list):
            for sub_item in item:
                key_value_pairs.append((sub_item, "TAG"))
        else:
            key_value_pairs.append((item, "TAG"))
    return key_value_pairs
r = []
for i,row in df.iterrows():
    f = format_to_key_value(df['Tags'][i])
    print(row['entities'])
    f = row['entities'] + f
    print(f)
    r.append(f)

df['entities'] = r

#%%
# Initialize a dictionary to store the graph information
graph = {"nodes": [], "links": []}

# Function to add a node to the graph
def add_node(node_name, entity_type, timestamp):
    for node in graph["nodes"]:
        if node["name"] == node_name and node["group"] == entity_type:
            if timestamp not in node["datetime"]:
                node["datetime"].append(timestamp)
                node["degree"] = len(node["datetime"])
                # Check if the degree is greater than 10 and add to graph nodes if it is
                if node["degree"] > 100:
                    if node not in graph["nodes"]:
                        graph["nodes"].append(node)
            return
    node = {
        "id": str(len(graph["nodes"]) + 1),
        "name": node_name,
        "group": entity_type,
        "degree": 1,
        "datetime": [timestamp],
    }
    graph["nodes"].append(node)

# Function to add a link to the graph
def add_link(source_node_id, target_node_id, timestamp):
    for link in graph["links"]:
        if link["source"] == source_node_id and link["target"] == target_node_id:
            if timestamp not in link["datetime"]:
                link["datetime"].append(timestamp)
            return
    link = {
        "source": source_node_id,
        "target": target_node_id,
        "datetime": [timestamp],
    }
    graph["links"].append(link)

# Iterate through the DataFrame to construct the co-occurrence graph
for idx, row in df.iterrows():
    timestamp = row['datetime']
    entities = row['entities']
    for entity, entity_type in entities:
        add_node(entity, entity_type, timestamp)
        for other_entity, other_entity_type in entities:
            if entity != other_entity:
                add_node(other_entity, other_entity_type, timestamp)
                # Find node IDs for the link
                source_node_id = [node["id"] for node in graph["nodes"] if node["name"] == entity and node["group"] == entity_type][0]
                target_node_id = [node["id"] for node in graph["nodes"] if node["name"] == other_entity and node["group"] == other_entity_type][0]
                add_link(source_node_id, target_node_id, timestamp)

# Convert graph to JSON format
json_output = json.dumps(graph, indent=2)

# Write JSON output to a file

with open("../vast3/public/keyword_graph.json", "w") as json_file:
    json_file.write(json_output)
