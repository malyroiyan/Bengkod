import pickle

# Simpan model terbaik
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)