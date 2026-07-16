import pandas as pd
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score
import os


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-folder', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for property in ["EE_before","EE_after","Aerosolization_Efficiency","mRNA_Recovery_Efficiency"]:
        pred = df[df["target"]==property]["y_pred"]
        true_value = df[df["target"]==property]["y_true"]
        mae = mean_absolute_error(pred,true_value)
        r2 = r2_score(pred,true_value)
        plt.plot(true_value, pred, 'o')
        plt.title("{}: MAE={:.4f}, R2={:.4f}".format(property, mae, r2))
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.savefig("{}/{}_scatter.png".format(args.output_folder, property))
        print("{}: MAE={:.4f}, R2={:.4f}".format(property, mae, r2))
        plt.close()