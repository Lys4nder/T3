from dataclasses import dataclass

from typing import Tuple, Optional

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.metrics import q_error
from src.model import FeatureMapper, TreeModel, PerTupleTreeModel, FlatTreeModel
from src.operators import OperatorType
from src.query_plan import QueryPlan
from src.training_data import build_per_tuple_training_data
from src.util import AutoNumber


class QueryCategory(AutoNumber):
    fixed = ()  # queries that are part of a benchmark and not generated
    select = ()  # table scans and selections
    pseudo_aggregate = ()  # table scans and aggregations (does not aggregate any values)
    aggregate = ()  # table scans and aggregations
    select_aggregate = ()  # table scans selections and aggregations
    join = ()  # multiple table scans and joins
    select_join = ()  # multiple table scans and joins
    join_agg = ()  # multiple table scans and joins with a group by at the end
    select_join_agg = ()
    join_simple_agg = ()  # multiple table scans and joins with a group by at the end but no groups
    select_join_simple_agg = ()
    complex_select = ()
    complex_select_agg = ()
    complex_select_join = ()
    complex_select_join_agg = ()
    complex_select_join_simple_agg = ()
    window = ()

    def get_name(self):
        names = {
            QueryCategory.fixed: "Fixed",
            QueryCategory.select: "Se",
            QueryCategory.pseudo_aggregate: "SiA",
            QueryCategory.aggregate: "A",
            QueryCategory.select_aggregate: "SeA",
            QueryCategory.join: "J",
            QueryCategory.select_join: "SeJ",
            QueryCategory.join_agg: "JA",
            QueryCategory.select_join_agg: "SeJA",
            QueryCategory.join_simple_agg: "JSiA",
            QueryCategory.select_join_simple_agg: "SeJSiA",
            QueryCategory.complex_select: "CSe",
            QueryCategory.complex_select_agg: "CSeA",
            QueryCategory.complex_select_join: "CSeJ",
            QueryCategory.complex_select_join_agg: "CSeJA",
            QueryCategory.complex_select_join_simple_agg: "CSeJSiA",
            QueryCategory.window: "W",
        }
        return names[self]


@dataclass
class BenchmarkedQuery:
    query_plan: QueryPlan
    total_runtimes: list[float]  # in seconds
    name: str
    query_text: str
    query_category: QueryCategory
    feature_matrix: Optional[np.ndarray] = None
    pipeline_runtimes: Optional[list[float]] = None

    def get_total_runtime(self) -> float:
        return np.median(self.total_runtimes)

    def get_analyze_plan_runtime(self) -> float:
        all_times = [x for p in self.query_plan.pipelines for x in (p.start, p.stop)]
        start = min(all_times)
        stop = max(all_times)
        # sometimes this will say 0, in this case we say 1 microsecond
        if start >= stop:
            return 1e-6
        return (stop - start) / 1_000_000

    def check_pipeline_overlap(self):
        pipelines = sorted(self.query_plan.pipelines, key=lambda p: (p.start, p.stop))
        for i, p in enumerate(pipelines[:-1]):
            p2 = pipelines[i + 1]
            if not p.stop <= p2.start:
                ids1 = [o.operator.op_id for o in p.operators]
                ids2 = [o.operator.op_id for o in p2.operators]
                common_ops = list(set(ids1).intersection(set(ids2)))
                if not len(common_ops) > 0:
                    print(" pipelines overlap without common op")
                common_ops = [o for o in p.operators if o.operator.op_id in common_ops]
                if len(common_ops) == 1 and common_ops[0].operator.type == OperatorType.SetOperation:
                    # SetOperation operators do not show pipeline correctly
                    # assert common_ops[0].operator.json["operation"] != "unionall", "unionall should be fixed already"
                    p.stop = p2.start
                    p2.stop = max(p.stop, p2.stop)
                else:
                    print("pipelines overlap!")
                    print(f" common ops: {', '.join(o.operator.operator_name for o in common_ops)}")

    def get_pipeline_runtimes(self, verbose: bool = False) -> list[float]:
        if self.pipeline_runtimes is None:
            total_time = self.get_total_runtime()
            analyze_plan_runtime = self.get_analyze_plan_runtime()
            # assert q_error(total_time, analyze_plan_runtime) < 2, "pipeline times seem off"
            if not q_error(total_time, analyze_plan_runtime) < 2 and total_time > 1e-5:
                if verbose:
                    print("pipeline times seem off")
            result = []
            self.check_pipeline_overlap()
            for p in self.query_plan.pipelines:
                result.append((p.stop - p.start) / (analyze_plan_runtime * 1_000_000) * total_time)
            pipeline_times_sum = sum(result)
            if abs(pipeline_times_sum - total_time) > max(total_time * 0.25, 0.0005):
                if verbose:
                    print(f"pipeline times do not add up: {pipeline_times_sum} != {total_time}")
            if pipeline_times_sum == 0:
                result = [total_time / len(result) for _ in result]
            else:
                correction_factor = total_time / pipeline_times_sum
                result = [x * correction_factor for x in result]
            pipeline_times_sum = sum(result)
            assert (
                abs(pipeline_times_sum - total_time) < 0.0005
            ), f"pipeline times do not add up: {pipeline_times_sum} != {total_time}"
            self.pipeline_runtimes = result
        return self.pipeline_runtimes

    def get_per_tuple_pipeline_runtimes(self) -> list[float]:
        result = []
        for pipeline, runtime in zip(self.query_plan.pipelines, self.get_pipeline_runtimes()):
            if pipeline.get_pipeline_scan_cardinality() == 0:
                result.append(runtime)
            else:
                result.append(runtime / pipeline.get_pipeline_scan_cardinality())
        return result

    def get_runtime_data(self, feature_mapper: FeatureMapper) -> Tuple[np.ndarray, float]:
        return feature_mapper.get_single_estimation_vector(self.query_plan), self.get_total_runtime()

    def get_pipeline_runtime_data(self, feature_mapper: FeatureMapper) -> list[Tuple[np.ndarray, float]]:
        features = feature_mapper.get_pipeline_estimation_matrix(self.query_plan)
        targets = self.get_pipeline_runtimes()
        return list((f, t) for f, t in zip(features, targets))

    def get_per_tuple_pipeline_runtime_data(self, feature_mapper: FeatureMapper) -> list[tuple[np.ndarray, float]]:
        features = self.get_feature_matrix(feature_mapper)
        targets = self.get_per_tuple_pipeline_runtimes()
        return list((f, t) for f, t in zip(features, targets))

    def get_feature_matrix(self, feature_mapper: FeatureMapper) -> np.ndarray:
        if self.feature_matrix is None:
            self.feature_matrix = feature_mapper.get_pipeline_estimation_matrix(self.query_plan)
        return self.feature_matrix


def optimize_tree_model(queries: list[BenchmarkedQuery], verbose: bool = False) -> TreeModel:
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []
    for query in queries:
        for x, y in query.get_pipeline_runtime_data(feature_mapper):
            x_vectors.append(x)
            y_values.append(y)
    x = np.vstack(x_vectors)
    y = np.array(y_values)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for _ in range(200):
        bst.update()
        if verbose:
            print(bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return TreeModel(bst)


def optimize_flat_tree_model(queries: list[BenchmarkedQuery], verbose: bool = False) -> FlatTreeModel:
    feature_mapper = FeatureMapper()
    x_vectors = []
    y_values = []
    for query in queries:
        current_x = []
        current_y = []
        for x, y in query.get_pipeline_runtime_data(feature_mapper):
            current_x.append(x)
            current_y.append(y)
        x_vectors.append(np.sum(current_x, axis=0))
        y_values.append(float(np.sum(current_y)))
    x = np.vstack(x_vectors)
    y = np.array(y_values)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for _ in range(200):
        bst.update()
        if verbose:
            print(bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return FlatTreeModel(bst)


def optimize_per_tuple_tree_model(queries: list[BenchmarkedQuery], verbose: bool = False) -> PerTupleTreeModel:
    feature_mapper = FeatureMapper()
    x, y = build_per_tuple_training_data(queries, feature_mapper, dtype=np.float64)
    seed = 21
    param = {"objective": "mape", "verbose": 2 if verbose else -1}
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    train_data = lgb.Dataset(x_train, label=y_train, feature_name=FeatureMapper.get_names(), params=param)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data, params=param)
    bst = lgb.Booster(param, train_data)
    bst.add_valid(val_data, "val_data")
    if verbose:
        print(bst.eval_train())
    for i in range(200):
        bst.update()
        if verbose:
            print(i + 1, bst.eval_train(), bst.eval_valid())
    if verbose:
        print(bst.eval_train(), bst.eval_valid())
    bst.save_model("model.txt")
    if verbose:
        for bench, y_true, y_pred in list(zip(queries, y, bst.predict(x))):
            print(f"{bench.name}: estimated time: {y_pred:.3f}, true time: {y_true:.3f}")
    return PerTupleTreeModel(bst)
