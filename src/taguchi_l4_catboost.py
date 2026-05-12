"""
Taguchi L4 Hyperparameter Optimization for CatBoost.
Runs 4 experiments using a 2-factor, 2-level design.
As per instructions, only 1 trial is used (Main Effects Plot for Means).
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
from src.compare_catboost import CatBoostPerTupleModel, _build_per_tuple_training_data

# 2 factors, 2 levels -> 4 experiments
L4_ARRAY = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
]

FACTORS = {
    'learning_rate': [0.05, 0.2],
    'depth': [4, 8],
}

def main():
    print("=== Taguchi L4 Optimization for CatBoost ===")
    
    predicted_cardinalities = False
    feature_mapper = FeatureMapper()
    
    print("Loading training data...")
    train_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities)
    x_full, y_full = _build_per_tuple_training_data(train_benchmarks, feature_mapper)
    
    print("Loading test data (All TPC-DS Test Queries)...")
    test_benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), predicted_cardinalities)
    
    results = []
    
    for i, row in enumerate(L4_ARRAY):
        exp_id = i + 1
        lr = FACTORS['learning_rate'][row[0]-1]
        dp = FACTORS['depth'][row[1]-1]
        
        print(f"\nExperiment {exp_id}/4: lr={lr}, depth={dp}")
        
        seed = 42 # Only 1 trial for L4
        x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, test_size=0.2, random_state=seed)
        
        reg = CatBoostRegressor(
            iterations=200,
            learning_rate=lr,
            depth=dp,
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
        
        print(f"  Result: Avg Q-Error = {avg_q_error:.4f}")
        
        results.append({
            'Experiment': exp_id,
            'LearningRate': lr,
            'Depth': dp,
            'Mean_QError': avg_q_error,
        })
        
    out_file = 'taguchi_catboost_l4_results.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nOptimization complete! Results saved to {out_file}")

if __name__ == '__main__':
    main()
