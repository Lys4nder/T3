"""
Taguchi L9 Hyperparameter Optimization for CatBoost.
Runs 9 experiments defined by the L9 Orthogonal Array.
For each experiment, it runs 3 trials (different random seeds) and records the Avg Q-Error on test set.
"""

import csv
import time
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.features import FeatureMapper
from src.metrics import q_error
from src.evaluation import QueryEstimationCache
from src.compare_catboost import CatBoostPerTupleModel
from src.training_data import build_per_tuple_training_data

# L9 Orthogonal Array (Rows=Experiments, Cols=Factors)
# Values are 1, 2, or 3 representing the level.
L9_ARRAY = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]

FACTORS = {
    'learning_rate': [0.05, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1.0, 3.0, 5.0],
    'border_count': [32, 64, 128],
}

def main():
    print("=== Taguchi L9 Optimization for CatBoost ===")
    
    predicted_cardinalities = False
    feature_mapper = FeatureMapper()
    
    print("Loading training data...")
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    x_full, y_full = build_per_tuple_training_data(train_benchmarks, feature_mapper)
    
    print("Loading test data (All TPC-DS Test Queries)...")
    test_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), predicted_cardinalities)
    
    results = []
    
    for i, row in enumerate(L9_ARRAY):
        exp_id = i + 1
        lr = FACTORS['learning_rate'][row[0]-1]
        dp = FACTORS['depth'][row[1]-1]
        l2 = FACTORS['l2_leaf_reg'][row[2]-1]
        bc = FACTORS['border_count'][row[3]-1]
        
        print(f"\nExperiment {exp_id}/9: lr={lr}, depth={dp}, l2_leaf_reg={l2}, border_count={bc}")
        
        trial_errors = []
        for trial, seed in enumerate([42, 100, 999], start=1):
            x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, test_size=0.2, random_state=seed)
            
            reg = CatBoostRegressor(
                iterations=200,
                learning_rate=lr,
                depth=dp,
                l2_leaf_reg=l2,
                border_count=bc,
                grow_policy="SymmetricTree",
                loss_function="RMSE",
                eval_metric="MAPE",
                random_seed=seed,
                thread_count=-1,
                verbose=False,
            )
            reg.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
            
            model = CatBoostPerTupleModel(reg)
            cache = QueryEstimationCache(model, predicted_cardinalities)
            
            estimates = [cache.queries[q.name].estimated_time for q in test_benchmarks]
            q_errors = [q_error(q.get_total_runtime(), est) for q, est in zip(test_benchmarks, estimates)]
            avg_q_error = np.mean(q_errors)
            
            trial_errors.append(avg_q_error)
            print(f"  Trial {trial} (seed {seed}): Avg Q-Error = {avg_q_error:.4f}")
            
        results.append({
            'Experiment': exp_id,
            'LearningRate': lr,
            'Depth': dp,
            'L2LeafReg': l2,
            'BorderCount': bc,
            'Trial1_Error': trial_errors[0],
            'Trial2_Error': trial_errors[1],
            'Trial3_Error': trial_errors[2],
        })
        
    out_file = 'taguchi_catboost_l9_results.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nOptimization complete! Results saved to {out_file}")

if __name__ == '__main__':
    main()
