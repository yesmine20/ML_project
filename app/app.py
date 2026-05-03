import os, sys, traceback
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from predict import predict_churn_from_form, load_artifacts, load_artifacts_rfm, load_artifacts_regression

app = Flask(__name__, template_folder='.')

# ── Chargement des modèles UNE SEULE FOIS au démarrage (performance) ──
print("[STARTUP] Chargement des modèles ML...")
try:
    _model, _imputer, _features = load_artifacts()
    _kmeans, _scaler_rfm, _imputer_rfm, _seg_map = load_artifacts_rfm()
    _model_reg, _metadata_reg = load_artifacts_regression()
    print("[STARTUP] Modèles chargés avec succès.")
except SystemExit as e:
    print(f"[STARTUP ERROR] Impossible de charger les modèles : {e}")
    print("                Lancez d'abord : python src/train_model.py")

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
    # FIX: debug=False en production (contrôlé par variable d'environnement)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)