from flask import Flask, render_template, request
import yfinance as yf
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('models/stock_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():

    stock = request.form['stock'].upper()

    try:
        # Fetch stock data
        data = yf.download(stock, period='5d')

        # Check if empty
        if data.empty:
            return render_template('index.html', error="Invalid Stock Symbol")

        # ---------------------------
        # SAFE CHART DATA
        # ---------------------------
        dates = data.index.astype(str).tolist()
        prices = data['Close'].values.flatten().tolist()

        # ---------------------------
        # SAFE SCALAR VALUE (FIX ALL ERRORS)
        # ---------------------------
        latest_price = data['Close'].squeeze().iloc[-1]
        latest_price = float(latest_price)

        # ---------------------------
        # PREDICTION
        # ---------------------------
        prediction = model.predict([[latest_price]])
        predicted_price = round(float(prediction[0]), 2)

        return render_template(
            'index.html',
            stock=stock,
            latest=round(latest_price, 2),
            prediction=predicted_price,
            dates=dates,
            prices=prices
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)