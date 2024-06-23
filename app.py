import numpy as np
from flask import Flask, request, render_template
import pickle
from configurations import DevelopmentConfig

# Create flask app
flask_app = Flask(__name__, template_folder='template',static_folder='static')

flask_app.config.from_object(DevelopmentConfig)

#loading files for boiler loss_prediction model
model = pickle.load(open("loss_prediction.pkl", "rb"))
std_scaler = pickle.load(open("final_scaler.pkl", "rb"))
t_scaler = pickle.load(open("target_scaler.pkl", "rb"))

#loading files for boiler efficiency_prediction model
model_be = pickle.load(open("model_loss.pkl", "rb"))
scaler_be = pickle.load(open("new_scaler.pkl", "rb"))


#build function to show home screen
@flask_app.route("/")
def Home():
    return render_template("home.html")

#build function to show boiler_efficiency model
@flask_app.route("/show_boiler_efficiency", methods = ["POST"])
def show_boiler_efficiency():
    return render_template("index_be.html")

#build function to show boiler_loss model
@flask_app.route("/show_boiler_loss", methods = ["POST"])
def show_boiler_loss():
    return render_template("index.html")
        


@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    features = std_scaler.transform(features)
    prediction = model.predict(features)
    prediction = t_scaler.inverse_transform(prediction)[0]
    p0 = prediction[0]
    p1 = prediction[1]
    p2 = prediction[2]
    p3 = prediction[3]
    p4 = prediction[4]
    p5 = prediction[5]
    
    
   
                          
    return render_template("index.html", p0=p0,p1=p1, p2=p2, p3=p3, p4=p4, p5=p5)
                           
                           
                           

if __name__ == "__main__":
    flask_app.run(debug=True)

