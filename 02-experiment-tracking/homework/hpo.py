import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_optimization(data_path: str, num_trials: int,server_uri: str = "http://127.0.0.1:5000",depth_from: int = 10, depth_to: int = 20,experiment_prefix: str = 'max_depth'):

    experiment_name = f'{experiment_prefix}_{depth_from}_{depth_to}'
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name)


    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run():
            mlflow.autolog()
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            run_id=mlflow.active_run().info.run_id
            # mlflow.log_metric('rmse', rmse)
        return {'loss': rmse, 'status': STATUS_OK, 'run_id': run_id}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', depth_from, depth_to, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }
    
    trials = Trials()
    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )
    best_result['run_id'] = trials.best_trial['result']['run_id']
    return best_result

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization_cli(**kwargs): print(run_optimization(**kwargs))

if __name__ == '__main__':
    run_optimization()
