import os, sys, traceback
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from predict import predict_churn_from_form

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.get_json()
        if form_data is None:
            return jsonify({'success': False, 'error': 'Corps JSON manquant'}), 400
        result = predict_churn_from_form(form_data)
        return jsonify({'success': True, **result})
    except Exception as e:
        # Imprime la traceback complète dans le terminal Flask
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 50)
    print("  Retail Churn Predictor — http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)