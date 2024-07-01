import pickle
import implicit
from implicit.evaluation import AUC_at_k, precision_at_k, mean_average_precision_at_k, ndcg_at_k
from scipy.sparse import load_npz
import os

train_csr_path = r"C:\Users\tup30579\Spotify Million\data_store\nameless_data\train_csr.npz"
test_csr_path = r"C:\Users\tup30579\Spotify Million\data_store\nameless_data\test_csr.npz"
train_csr = load_npz(train_csr_path)
test_csr = load_npz(test_csr_path)

ALS_model = implicit.als.AlternatingLeastSquares()
BPZ_model = implicit.bpr.BayesianPersonalizedRanking()
LMF_model = implicit.lmf.LogisticMatrixFactorization()

ALS_save_path = r"C:\Users\tup30579\Spotify Million\results\AlternatingLeastSquares\ALS_model.pkl"
BPZ_save_path = r"C:\Users\tup30579\Spotify Million\results\BayesianPersonalizedRanking\BPZ_model.pkl"
LMF_save_path = r"C:\Users\tup30579\Spotify Million\results\LogisticMatrixFactorization\LMF_model.pkl"

def Pipeline(model, train_data, test_data, save_path):
    model.fit(train_data)
    print("Model Trained!")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    precision = precision_at_k(model, train_data, test_data, K=10)
    print("Precision Calculated!")
    return(precision)

ALS_precision = Pipeline(model = ALS_model,train_data = train_csr,test_data = test_csr,save_path = ALS_save_path)
print(f"For ALS, precision: {ALS_precision}")
BPZ_precision = Pipeline(model =BPZ_model,train_data = train_csr,test_data = test_csr,save_path = BPZ_save_path)
print(f"For BPZ, precision: {BPZ_precision}")
LMF_precision = Pipeline(model =LMF_model,train_data = train_csr,test_data = test_csr,save_path = LMF_save_path)
print(f"For LMF, precision: {LMF_precision}")

