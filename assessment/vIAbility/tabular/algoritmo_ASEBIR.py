import argparse
import os, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET

implantation = {
    'hcg testpositiv': 1,
    'pos': 1,
    'positiv': 1,
    'biokjemisk': 1,       
    'neg': 0,
    'negativ': 0,
    'negative': 0,
    'negativ (står positiv i lf uten mer informasjon, men neg. i sys.x.)': 0
}

def implantation_probability(t2, cc2, cc3, s2):
    if cc2 <= 5:
        return 0.0
    else:
        if s2 >= 5:
            return 12.50
        else:
            t2_condition = t2 >= 24 and t2 <= 25
            if cc3 < 11 or cc3 >15:
                if t2_condition:
                    return 40.0
                else:
                    return 29.40
            else:
                if t2_condition:
                    return 70.40
                else:
                    return 48.70


def algorithm_as_classificator(dataset_path,file_path = None, column = "Fetal Heart Beat", threshold = 50):
    """
    dataset_path = "../Datasets/Processed/Oslo_viability/ORIGINAL.xlsx"
    column = "Fetal Heart Beat" 
    column = "HCG Test"
    """
   
    df = pd.read_excel(dataset_path, sheet_name="Morula_Blast_nov")
    if file_path:
        test_df = pd.read_csv(file_path)
        test_df['id'] = test_df['filename'].apply(lambda x: Path(x).stem.split("_Well_")[0])
        test_df['well'] = test_df['filename'].apply(lambda x: int(Path(x).stem.split("_Well_")[1]))
        valid_pairs = set(zip(test_df['id'], test_df['well']))
        df = df[df.apply(lambda row: (row['Slide ID'], row['Well']) in valid_pairs, axis=1)]

    df = df.dropna(subset=['HCG Test', "t2", "t3", "t4", "t5", "t6"])
    df = df.reset_index(drop=True)
    if column == "Fetal Heart Beat":
        df[column] = df[column].fillna(0)
        df= df[df[column]!='mi']
    if column == "HCG Test":
        df['HCG Test'] = df['HCG Test'].map(implantation)
    df['cc2'] = df['t4'] - df['t2']
    df['cc3'] = df['t6'] - df['t3']
    df['s2']  = df['t5'] - df['t3']

    df['Implantation Probability'] = df.apply(lambda row: implantation_probability(row['t2'], row['cc2'], row['cc3'], row['s2']), axis=1)
    print("Accuracy:", accuracy_score(df[column]>0, df['Implantation Probability'] > threshold))
    print(classification_report(df[column]>0, df['Implantation Probability'] > threshold, zero_division=0))
    print("Confusion Matrix:")
    cm = confusion_matrix(df[column]>0, df['Implantation Probability'] > threshold)
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(f"confusion_matrix_{column}.png", dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASEBIR algorithm as classificator")
    parser.add_argument("--dataset_path", help="Path to the dataset file")
    parser.add_argument("--file_path", help="Path to the CSV file containing test video filenames")
    parser.add_argument("--threshold", type=float, help="Threshold for classification")
    args = parser.parse_args()
    dataset_path = args.dataset_path or os.path.join(PATH_DATASET, "Oslo_viability/original.xlsx")
    file_path = os.path.join(PATH_DATASET, "Oslo_viability/", args.file_path)
    threshold = args.threshold or 50
    
    print("Classification report for Fetal Heart Beat:")
    algorithm_as_classificator(dataset_path, file_path=file_path, column = "Fetal Heart Beat", threshold = threshold)
    print("\nClassification report for HCG Test:")
    algorithm_as_classificator(dataset_path, file_path=file_path, column = "HCG Test", threshold = threshold)