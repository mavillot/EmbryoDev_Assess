import os, json
import pandas as pd
from pathlib import Path

def main(args):
    dir = Path("results/assessment")/args.assess / "features" / "Combined" / args.feat / args.model / args.loss
    filename = f"{args.assess}_{args.feat}_{args.model}_{args.loss}.csv"
    for i in range(1, 6):
        metrics = dir / f"iter{i}" / "final_metrics.json"
        run_config = dir / f"iter{i}" / "run_config.json"
        if not metrics.exists():
            print(f"Missing {metrics}")
            continue
        with open(metrics, "r") as f:
            data_metrics = json.load(f)
        with open(run_config, "r") as f:
            data_config = json.load(f)
            #only keep the relevant keys: loss, iteration, dataset, feat, pos_enc
            data_config = {k: v for k, v in data_config.items() if k in ["iteration", "feat", "model","loss", "use_pos_encoding"]}
        data = {**data_config, **data_metrics}
        df = pd.DataFrame([data])
        if i == 1:
            all_df = df.copy()
        else:
            all_df = pd.concat([all_df, df], ignore_index=True)
    all_df.to_csv(filename, index=False)

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Aggregate results from multiple iterations into a single CSV")
    # parser.add_argument("--assess", type=str, default="ploidIA", choices=["ploidIA","vIAbility"])
    # parser.add_argument("--feat", type=str, default="resnet34")
    # parser.add_argument("--model", type=str, default="gru")
    # parser.add_argument("--loss", type=str, default="ce")
    # args = parser.parse_args()
    feat = ["markers"]
    model = ["gru", "lstm", "rnn", "transformer"]
    loss = ["fbeta"]
    for f in feat:
        for m in model:
            for l in loss:
                print(f"Aggregating results for feat={f}, model={m}, loss={l}")
                args = type("Args", (object,), {"assess": "ploidIA","feat": f, "model": m, "loss": l})()
                main(args)
