import subprocess

from flask import Flask, request
import json
import openai
import os
import io

app = Flask(__name__)
app.config.update(SERVER_NAME='130.82.27.169:5000')

img_path = 'web/image.jpg'
graph_path = 'web/graph.json'

openai.api_key = os.getenv("OPENAI_API_KEY")
language_model = "text-davinci-001"

topk = 5


@app.route('/upload', methods=['POST'])
def get_scenedescription():
    with open(img_path, "wb") as file:
        file.write(io.BytesIO(request.data).getvalue())
        file.close()
    generate_scene_graph(".", img_path, graph_path, device="cpu", topk=topk)
    triples = get_triples(graph_path)
    scene_description = transform_to_sentence(triples)
    return {"description": scene_description}


def transform_to_sentence(triples):
    gpt_query = "Describe the following triples in a nice story scene setting: "
    triples_string = ""
    for t in triples:
        triples_string += t + " "
    gpt_query += triples_string[:-1] # remove trailing space
    print("Sending query for ", triples_string)
    response = openai.Completion.create(engine=language_model, prompt=gpt_query, max_tokens=200)
    answer = response.choices[0].text
    return answer.replace("\n", "")


def get_triples(graph_path):
    """
    Complete import of a graph created with RelTR
    """
    triples_read = []
    with open(graph_path, "r") as file:
        triples = json.load(file)
        file.close()

    for triple_dict in triples:
        subject = triple_dict["subject"]["id"]
        predicate = triple_dict["predicate"]["id"]
        object = triple_dict["object"]["id"]
        triples_read.append(f"({subject}, {predicate}, {object})")
    print(f"Loaded {len(triples_read)} objects from {graph_path}")
    return triples_read


def generate_scene_graph(reltr_path, img_path, graph_path, device="cpu", topk=5):
    """
    calls RelTR to create scene graph from image and saves json output file in graph path
    """
    subprocess.check_output([f'python',
                             f"{reltr_path}/mkgraph.py",
                             "--img_path", f"{img_path}",
                             "--device", f"{device}",
                             "--resume", f"{reltr_path}/ckpt/checkpoint0149.pth",
                             "--export_path", f"{graph_path}",
                             "--topk", f"{topk}"])


def test_get_triples():
    test_graph_path = "web/test/graph.json"
    triples = get_triples(test_graph_path)
    for t in triples:
        print(t)


def test_get_desc():
    test_graph_path = "web/test/graph.json"
    triples = get_triples(test_graph_path)
    scene_description = transform_to_sentence(triples)
    print(scene_description)


def test_reltr():
    test_img_path = "web/test/desk.jpg"
    generate_scene_graph(".", test_img_path, graph_path, device="cpu", topk=topk)
    triples = get_triples(graph_path)
    for t in triples:
        print(t)


if __name__ == "__main__":
    app.run(debug=False)
    # test_get_triples()
    # test_get_desc()
    # test_reltr()
