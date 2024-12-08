from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        if not news_text:
            return render_template('result.html', prediction="No text provided!")

        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]

        # Render result
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

