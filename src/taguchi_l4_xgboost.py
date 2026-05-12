"""
Taguchi L4 Hyperparameter Optimization for XGBoost.
Runs 4 experiments using a 2-factor, 2-level design.
Runs multiple seeds per experiment and aggregates results.
"""

import csv
import time
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.features import FeatureMapper
from src.metrics import q_error
from src.evaluation import QueryEstimationCache
from src.compare_xgboost import XGBPerTupleModel, _build_per_tuple_training_data

# 2 factors, 2 levels -> 4 experiments
L4_ARRAY = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
]

FACTORS = {
    'learning_rate': [0.05, 0.2],
    'max_depth': [4, 12],
}

# 5 trials (seeds) per experiment for stability
SEEDS = [1,2,3,4,5]

def main():
    print("=== Taguchi L4 Optimization for XGBoost ===")
    
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
        md = FACTORS['max_depth'][row[1]-1]
        
        print(f"\nExperiment {exp_id}/4: lr={lr}, max_depth={md}")
        
        per_seed_q_errors = []

        for seed in SEEDS:
            x_train, x_val, y_train, y_val = train_test_split(
                x_full,
                y_full,
                test_size=0.2,
                random_state=seed,
            )

            reg = XGBRegressor(
                n_estimators=200,
                learning_rate=lr,
                max_depth=md,
                tree_method='hist',
                subsample=0.8,
                grow_policy="lossguide",
                max_leaves=31,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                eval_metric="mape",
                n_jobs=-1,
                random_state=seed,
            )
            reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

            model = XGBPerTupleModel(reg)
            cache = QueryEstimationCache(model, predicted_cardinalities)

            estimates = [cache.queries[q.name].estimated_time for q in test_benchmarks]
            q_errors = [q_error(q.get_total_runtime(), est) for q, est in zip(test_benchmarks, estimates)]
            per_seed_q_errors.append(float(np.mean(q_errors)))

        mean_q_error = float(np.mean(per_seed_q_errors))
        std_q_error = float(np.std(per_seed_q_errors, ddof=1)) if len(per_seed_q_errors) > 1 else 0.0

        print(f"  Result: Mean Q-Error = {mean_q_error:.4f} (std={std_q_error:.4f}, seeds={len(SEEDS)})")
        
        results.append({
            'Experiment': exp_id,
            'LearningRate': lr,
            'MaxDepth': md,
            'Mean_QError': mean_q_error,
            'Std_QError': std_q_error,
            'NumSeeds': len(SEEDS),
        })
        
    out_file = 'taguchi_xgboost_l4_results.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nOptimization complete! Results saved to {out_file}")

if __name__ == '__main__':
    main()
