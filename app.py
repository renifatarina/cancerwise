from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    str_pred=''
    pred = 0
    if request.method == "POST":
        # Collect input data from the form
        gender = int(request.form["gender"])
        age = float(request.form["age"])
        age = (age - 63.014) / 43.499
        yellow_finger = int(request.form["yellow_finger"])
        anxiety = int(request.form["anxiety"])
        peer_pressure = int(request.form["peer_pressure"])
        chronic_disease = int(request.form["chronic_disease"])
        fatigue = int(request.form["fatigue"])
        allergy = int(request.form["allergy"])
        wheezing = int(request.form["wheezing"])
        alcohol_consuming = int(request.form["alcohol_consuming"])
        coughing = int(request.form["coughing"])
        swallowing_difficulty = int(request.form["swallowing_difficulty"])
        chest_pain = int(request.form["chest_pain"])

        list_data = [gender, age, yellow_finger, anxiety, peer_pressure, chronic_disease, fatigue, allergy,
                     wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain]

        prediction = model.predict([list_data])
        prediction = prediction[0,0]
        # Convert the prediction to a human-readable result
        if prediction >= 0.5:
            str_pred='Mohon maaf, Anda mengalami gejala kanker paru-paru.'
            pred=1
        elif prediction < 0.5:
            str_pred='Kabar baik! Anda tidak menderita kanker paru-paru.'
            pred=0

    return render_template("index.html", str_pred=str_pred, pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
