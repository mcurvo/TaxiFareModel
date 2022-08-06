import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from TaxiFareModel.trainer import Trainer
from TaxiFareModel.data import get_data

EXPERIMENT_NAME = "[UK] [London] [mcurvo] Murilo Curvo 1.0"

MLFLOW_URI = "https://mlflow.lewagon.ai/"
# Indicate mlflow to log to remote server
mlflow.set_tracking_uri(MLFLOW_URI)

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")

yourname = "Murilo Curvo"

df = get_data(n_rows=100_000)

# set X and y
X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

# hold out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

test_pipe = Trainer(X_train, X_test, y_train, y_test)

# build pipeline
pipeline = test_pipe.set_pipeline()

# train the pipeline
test_pipe.run(X_train, X_test, y_train, y_test, pipeline)

# evaluate the pipeline
y_pred = test_pipe.evaluate
rmse = evaluate( pipeline)





if yourname is None:
    print("please define your name, il will be used as a parameter to log")

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
