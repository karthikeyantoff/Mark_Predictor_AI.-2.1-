from flask import Flask, render_template, request
import torch
import torch.nn as nn

app = Flask(__name__)

# Load model config
config = torch.load("pytorch_project/csv_folder/model_config.pth")
activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "silu": nn.SiLU(),
}
activation = activation_map[config["activation"]]

# Load model
model = nn.Sequential(
    nn.Linear(3, config["hidden1"]),
    activation,
    nn.Linear(config["hidden1"], config["hidden2"]),
    activation,
    nn.Linear(config["hidden2"], 1)
)
model.load_state_dict(torch.load("pytorch_project/csv_folder/pytorch_traindata.pth"))
model.eval()

# Prediction function
def predict(inputs):
    x = torch.tensor([inputs], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
    marks = output[0][0].item()
    return round(marks, 2)

# Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            StudentID=int(request.form["StudentID"])
            StudentName=request.form["StudentName"]
            StudentCourse=request.form["StudentCourse"]
            StudHours= float(request.form["StudHours"])
            SleepHours = float(request.form["SleepHours"])
            PlayHours= float(request.form["PlayHours"])
            Marks = predict([StudHours, SleepHours, PlayHours])
            result = f"Student id:{StudentID},Student Name:{StudentName},Student Courses:{StudentCourse}, AND predicted Marks:{Marks}"
        except:
            result = "Invalid input. Please enter numbers only."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
