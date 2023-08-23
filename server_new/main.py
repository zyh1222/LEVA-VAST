# %%
from flask import Flask, request, jsonify
import pandas as pd
from get_messages import *
import json
from flask_cors import CORS
import openai
import re
from util import load_json, load_jl, save_jl_append, save_txt
from similar_docs import get_similar_docs
import os
import time
from datetime import datetime, timedelta
from insights import *

app = Flask(__name__)
CORS(app)

# openai.api_key = "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"

openai.api_key = "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"

step = 1
round = 1

hist_recommends = []
hist_inter = []
insights_hist_inter = []
hist_inter_sequences = []
latex_hist_inter = []

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s
  
def get_messages_by_id(ID_list):
    # 从JSON文件中读取数据，并转换为DataFrame
    df = pd.read_json("../vast3/public/data.json")

    # 根据提供的ID列表过滤DataFrame，并获取对应的id和message列
    filtered_data = df[df['ID'].isin(ID_list)][['ID', 'message']]

    # 将DataFrame转换为字典的列表，每个字典形式为{id: message}
    messages_list = filtered_data.to_dict(orient='records')

    return messages_list

def get_messages_by_timerange(time_range):
    # 从JSON文件中读取数据，并转换为DataFrame
    df = pd.read_json("../vast3/public/data.json")

    # 根据提供的ID列表过滤DataFrame，并获取对应的id和message列
    filtered_data = df[(df['datetime'] >= time_range[0]) & (df['datetime'] <= time_range[1])]
    filtered_data = filtered_data.sort_values(by="datetime")
    return filtered_data
  
def get_messages_by_subgraph(graph, timerange):
    start_time = time_to_seconds(timerange[0])
    end_time = time_to_seconds(timerange[1])
    
    subgraph_nodes = graph["nodes"]
    subgraph_links = graph["links"]
    # Initialize an empty list to store matching rows
    matching_rows = []
    # Iterate through the original DataFrame to find matching text
    df = get_messages_by_timerange([start_time,end_time])
    
    for idx, row in df.iterrows():
        timestamp = row['datetime']
        entities = row['Entities']
        tags = row['Tags']
        # Check if the current row contains entities and tags from the subgraph
        contains_entities = all(any(node["name"] == entity and node["group"] == entity_type for entity, entity_type in entities) for node in subgraph_nodes)
        contains_tags = all(tag in tags for tag in tags)  # Assuming you have subgraph_tags defined

        if contains_entities or contains_tags:
            matching_rows.append({
                "datetime": timestamp,
                "text": row['text'],
                "ID" : row['ID']
            })
    # Create a DataFrame from the matching rows
    matching_df = pd.DataFrame(matching_rows)
    # Print the resulting DataFrame
    print(matching_df)
    matching_df.set_index("datetime", inplace=True)
    return matching_df


def get_messages_df_by_id(ID_list):
    # 从JSON文件中读取数据，并转换为DataFrame
    df = pd.read_csv("./assets/data/data.csv")

    # 根据提供的ID列表过滤DataFrame，并获取对应的id和message列
    filtered_data = df[df['ID'].isin(ID_list)][['datetime', 'message', 'ID']]

    return filtered_data


def chat(text):
    completions = openai.ChatCompletion.create(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text},
        ]
    )
    message = completions.choices[0].message
    return message["content"]

def askChatGPT(messages):
    completions = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature = 0
    )
    return completions.choices[0].message["content"]
  

@app.route('/api/msg', methods=['GET'])
def msg():
    msg = request.args.get('params')
    # res = chat(msg)
    # 历史记录list + 所有数据 + 当前数据子集 -》 判断需要所有的还是当前即可
    print(msg)
    res = "11111111111111"
    time.sleep(5)
    return res
  
def get_questions():
    with open("./apitest.txt", "r") as f:
        txt = f.readlines()
    messages = [{"role": "user", "content": txt}]



def get_response(file_path,keyword):
    
    with open(file_path, "r") as f:
        data = f.read()

    prompt = f"""I have a data table. I need you to summarize the below messages of the keyword "{keyword}" into a story with several subevents. You can try to figure out when and where the subevent is.  At the end of your answer, as each subevent may consist of one or more messages, you should use a list (e,g.,["1", "2",...]) to list the "ID" of the message. Use the messages as a source of facts, and do not engage in unfounded speculation.  Subevent Title no more than 4 words. The output format:""" + """\n
    {
        "Story Title": short title,
        "Subevents": [
            {
            "Subevent": short title,
            "Location": "",
            "Time": "start-end",
            "Messages": [],
            "Summary": ""
            }
        ]
    })""" + f"""\nDo not add other sentences.\n Data: {data} \n"""
    res = chat(prompt)
    json_response = json.loads(res)
    save_jl_append(json_response, f"./assets/output/story_explore{keyword}_summary.json")
    return json.dumps(json_response)


@app.route('/api/map_data', methods=['POST'])
def map_data():
    selected_data = request.get_json()
    print(selected_data)
    return jsonify({'message': 'map data received'}), 200

def match_filenames_with_time_range(target_time_range, folder_path):
    matched_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_start_time_str, file_end_time_str = filename[:-6].split('-')
            file_start_time = datetime.strptime(file_start_time_str, "%H:%M")
            file_end_time = datetime.strptime(file_end_time_str, "%H:%M")

            start_time_difference = abs((target_time_range[0] - file_start_time).total_seconds())
            end_time_difference = abs((target_time_range[1] - file_end_time).total_seconds())

            if start_time_difference <= 60 and end_time_difference <= 60:  # 3分钟内的时间差
                matched_files.append(os.path.join(folder_path, filename))

    return matched_files

@app.route('/api/bins_risk', methods=['POST'])
def bins_risk():
    start = time.time()
    data = request.get_json()
    time_range = data[0]
    bins = data[1]
    bins = pd.DataFrame(bins.values())  
    print(list(bins['risk']))
    topN_index = get_outstanding_topn(list(bins['risk']))
    if len(topN_index) > 3:
      topN_index = topN_index[:3]
    print(topN_index, bins['risk'].iloc[topN_index])
    return {"bins_index":[str(i) for i in topN_index]}
    
    
@app.route('/api/bins', methods=['POST'])
def bins():
    # 估计程序执行时间
    start = time.time()
    data = request.get_json()
    time_range = data[0]
    bins = data[1]
    non_geo = data[2]

    folder_path = "./assets/output/summary/select_time/"
    strptime_range = [datetime.strptime(time_range[0], "%H:%M"), datetime.strptime(time_range[1], "%H:%M")]
    matched_files = match_filenames_with_time_range(strptime_range, folder_path)
    print(matched_files)
    if len(matched_files) > 0 and len(load_jl(matched_files[0])):
      time.sleep(4)
      return load_jl(matched_files[0])[-1]
    else:
      print("no matched files")
      
      prompt = f"""I have a dataset that contains multiple regions, and each region has some messages that must be published in that region and some semantically relevant messages. Based on those messages, I need you to summarize an event story for each region. You can try to figure out when and where the event is. The output format:""" + """
      {
          "Events": [
              {
              "Event": short title,
              "Location": "",
              "Time": "",
              "Messages": [],
              "Summary": ""
              }
          ]
      })""" + f"""\n At the end of your answer, as each event may consist of one or more messages, you should use a list (e,g.,["1", "2",...]) to list the "ID" of the Messages. Use the messages as a source of facts, and do not engage in unfounded speculation. Event Title no more than 4 words. "Location" might be empty but "Summary" should not be empty. "Summary" no more than 20 words. Do not add other sentences.\n"""
      
      bins = pd.DataFrame(bins.values())  
      print(list(bins['risk']))
      topN_index = get_outstanding_topn(list(bins['risk']))
      if len(topN_index) > 4:
        topN_index = topN_index[:4]
      print(topN_index, bins['risk'].iloc[topN_index])
      num = len(topN_index)
      save_path = f"./assets/data/select_time/{time_range[0]}-{time_range[1]}.txt"
      prompt_save_path = f"./assets/input/select_time/{time_range[0]}-{time_range[1]}.txt"
      
      j = 0
      for i,row in bins.iloc[topN_index].iterrows():
          j += 1
          df_search = get_messages_df_by_id(list(row['IDs']))
          if len(df_search) < int(100/num):
              k = int(100/num)-len(df_search)
          else:
              continue
          df_search_context = get_messages_df_by_id(non_geo)

          result = get_similar_docs(df_search_context=df_search_context, question='\n'.join(row['message']), k=k)
          df_context = pd.concat([df_search, result])
          df_context = df_context.sort_values(by="ID")
          df_context.set_index("datetime", inplace=True)
          df_context.to_csv(save_path)
          
          with open(save_path, 'r') as file:
              context = file.read()
              
          prompt += f"""Event{j}:\n {context} """
      save_txt(prompt, prompt_save_path)
      res = chat(prompt)
      json_response = {"bins_index":[str(i) for i in topN_index], "data":json.loads(res)}
      print(json_response)
      save_jl_append(
          json_response, f"./assets/output/summary/select_time/{time_range[0]}-{time_range[1]}.jsonl")
      
      endtime = time.time()
      print("calculate multiple events time:", endtime-start)
      return json_response

@app.route('/api/graph_interpret', methods=['POST'])
def graph_interpret():
    folder_path = "./assets/output/summary/select_time/interpret/"
    data = request.get_json()
    time_range = data[0]
    graph = data[1]
    messages = get_messages_by_subgraph(graph, time_range)
    prompt  = """I have a graph and some messages. I need you to analyze the below messages data and graph data to interpret the subgraph means. The output format: """
    res = chat(prompt)
    print(res)
  
@app.route('/api/graph_data', methods=['POST'])
def graph_data():
    folder_path = "./assets/output/summary/select_time/keyplayer/"
    data = request.get_json()
    time_range = data[0]
    time_Annotation_selected = data[1]
    graph = data[2]
    messages = []
    for obj in time_Annotation_selected:
        messages.append(get_messages_by_id(obj['Messages']))
    prompt = f"""The analyst is interested in these events you summarized: \n {time_Annotation_selected}. \nBy processing the messages supporting above events, we build a graph by extrating tags, person, organizations and locations. Now I need you to analyze the below messages data and graph data to find nodes in the graph related to the events above. For each event, find the corresponding message according to its id as the supplementary information for you to understand this event, and then find out the id of nodes related to this event according to the node name from graph data. Each event finds no more than three nodes. The output format: """ + \
        """{"Related_nodes":[{"Event": Event title in above list,"nodeID":},{"Event":,"nodeID":}]}. Do not add other sentences.""" + \
             f"""\n Messages data: \n{messages}\n Graph data:\n{graph}"""
    save_txt(prompt, f"./assets/input/{time_range[0]}-{time_range[1]}_graph_annotation.txt")
    if os.path.exists(f'./assets/output/summary/select_time/keyplayer/{time_range[0]}-{time_range[1]}.jsonl'):
        time.sleep(5)
        return load_jl(f'./assets/output/summary/select_time/keyplayer/{time_range[0]}-{time_range[1]}.jsonl')[-1]
    res = chat(prompt)
    print(res)
    json_response = json.loads(res)
    save_jl_append(
        json_response, f"{folder_path}{time_range[0]}-{time_range[1]}.jsonl")  
    # json_response = json.dumps(
    #     {"Related_nodes":[{"Subevent": "Van Hits a Biker", "nodeID": 131}]})

    # pattern = r'\[(\s*\d+(?:\s*,\s*\d+)*)\]'
    # matches = re.findall(pattern, res)
    # all_numbers_str = ''.join(matches).replace(' ', '')
    # all_numbers_list = [int(num) for num in all_numbers_str.split(',')]
    # all_numbers_json = json.dumps(all_numbers_list)
    # all_numbers_json = json.dumps([131])
    # time_range = [datetime.strptime(time_range[0], "%H:%M:%S"), datetime.strptime(time_range[1], "%H:%M:%S")]
    # matched_files = match_filenames_with_time_range(time_range, folder_path)
    return json_response


@app.route('/api/keyword', methods=['POST'])
def keyword():
    global keyword
    keyword = request.get_data(as_text=True)
    file_path = f'./assets/data/{keyword}.txt'
    if not os.path.exists(file_path):
        df_all = get_messages(keyword, topn = 1)
    if os.path.exists(f'./assets/output/summary/select_keyword/{keyword}_summary.json'):
        time.sleep(5)
        return load_json(f'./assets/output/summary/select_keyword/{keyword}_summary.json')

    result = get_response(file_path, keyword)
    # result = load_json('./assets/data/message_summary.json')
    return result

@app.route('/api/comparison', methods=['POST'])
def comparison():
    return{'keyword': "van", "summary":"a black van is mentioned in multiple events: 'Hit and Run', 'Shooting at Gelatogalore'."}


@app.route('/api/timeline_data', methods=['POST'])
def timeline_data():
    selected_data = request.get_json()
    # print(selected_data)
    return jsonify({'message': 'timeline data received'}), 200


@app.route('/api/location_data', methods=['POST'])
def location_data():
    selected_data = request.get_json()
    # print(selected_data)
    return jsonify({'message': 'map data received'}), 200


@app.route('/api/risk_data', methods=['POST'])
def risk_data():
    selected_data = request.get_json()
    # print(selected_data)
    return jsonify({'message': 'risk data received'}), 200

if __name__ == '__main__':
    app.run(debug=True)
