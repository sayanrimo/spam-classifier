from flask import Flask, request, render_template
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form["email_text"]
        text_vector = vectorizer.transform([email_text])
        prediction = model.predict(text_vector)[0]
        result = "Spam" if prediction == 1 else "Ham"
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
