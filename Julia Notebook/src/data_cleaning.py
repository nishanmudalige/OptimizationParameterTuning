import pandas as pd
import os

RESULT_PATH = os.path.join("./", "Julia Notebook", "results")

lbfgs_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_lbfgs.csv"))
scalable_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_scalable.csv"))

lbfgs_df["is_scalable"] = False
scalable_df["is_scalable"] = True
new_df = pd.concat([lbfgs_df, scalable_df])
new_df.to_csv(os.path.join(RESULT_PATH, "complete_dataset_as_of_Nov6.csv"), index=False)

