from flask import Flask, request, jsonify, Response
from pipeline import PipeLine

app = Flask(__name__)
pipeline_obj = PipeLine()

@app.route("/", methods=["GET"])
def loadbalancer_check():
    return Response("OK", status=200)


@app.route("/predict", methods=["POST"])
def main():
    """
    predict the class based on input text 
    """
    
    requestData = request.get_json(force=True)
    
    prediction = pipeline_obj.predict(requestData["input"])
    
    output = {
        "fine_tuned_prediction": prediction
    }
    return jsonify(output)

@app.route("/benchmark", methods=["GET"])
def benchmarking():
    """
    Based on model name and test set gives accuracy of the model to benchmark both the model performances 

    """
        
    accuracy = pipeline_obj.benchmark_pipeline()
    
    output = {
        "pretrained_benchmark_accuracy": accuracy[1],
        "finetune_benchmark_accuracy": accuracy[0]
        
    }
    return jsonify(output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)