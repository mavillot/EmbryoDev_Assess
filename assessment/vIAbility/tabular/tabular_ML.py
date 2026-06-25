import numpy as np
import pandas as pd
import pprint
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score, accuracy_score, f1_score

# Classifiers used in your pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- 1. DATASET CONFIGURATION ---
df = pd.read_csv("assessment/vIAbility/tabular/tabular_data.csv")  
df.dropna(subset=['Fetal Heart Beat'], inplace=True)
X = df.drop(columns=['Fetal Heart Beat', 'HCG Test'])
y = df['Fetal Heart Beat']


# Defining 5 different seeds to evaluate consistency (equivalent to your 5 runs in DL)
SEEDS = [42, 101, 2023, 777, 999]
N_REPETITIONS = len(SEEDS)

# --- 2. CLASSIFIERS AND HYPERPARAMETER SPACE DEFINITION ---
classifiers_config = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 4, 6, 8, 12],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    'SVC': {
        'model': SVC(probability=True, random_state=42), # probability=True is needed for ROC-AUC / prediction thresholds
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [3, 5, 7, 11, 15],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        }
    },
    'GradientBoosting': {
        'model': HistGradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_iter': [100, 150],
            'classifier__max_depth': [3, 5, 7]
        }
    },
    'TabPFN': {
        'model': TabPFNClassifier(ignore_pretraining_limits=True),
        'params': {}
    }
}

# History dictionary to store all test metrics across repetitions
results_history = {clf_name: {'accuracy': [],"precision": [],'recall': [], 'f1_macro': [], 'fbeta': [],'roc_auc': [],} for clf_name in classifiers_config}

print("=== Starting ML Models Consistency Study (5 Independent Seeds) ===")

# --- 3. CONSISTENCY STUDY LOOP ---
for i, seed in enumerate(SEEDS):
    print(f"\n▶ Run {i+1}/{N_REPETITIONS} (Global Seed / Data Partition: {seed})")
    
    # Split the dataset using the current iteration seed to introduce sample variance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=seed, 
        stratify=y
    )
    
    # Inner CV strategy to find the best hyperparameters within the current training data fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for clf_name, config in classifiers_config.items():
        # Set internal seed for stochastic algorithms if applicable
        if 'random_state' in config['model'].get_params():
            config['model'].set_params(random_state=seed)
            
        # Build standard ML Pipeline (Scaler + Classifier) to prevent data leakage
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config['model'])
        ])
        
        # Grid Search to tune parameters on Train Data
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Training/Fitting
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluation on the independent Test subset for this specific seed
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics Computation
        auc_score = roc_auc_score(y_test, y_pred_proba)
        acc_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        fbeta = fbeta_score(y_test, y_pred, beta=1.0, zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        
        # Storing individual scores
        results_history[clf_name]['roc_auc'].append(auc_score)
        results_history[clf_name]['accuracy'].append(acc_score)
        results_history[clf_name]['f1_macro'].append(f1_mac)
        results_history[clf_name]['precision'].append(precision)
        results_history[clf_name]['recall'].append(recall)
        results_history[clf_name]['fbeta'].append(fbeta)
        
        print(f"   - {clf_name:20} -> Test ROC-AUC: {auc_score:.4f} | Accuracy: {acc_score:.4f}")

# --- 4. SUMMARY STATISTICS & FINAL MATRIX REPORT ---
print("\n" + "="*60)
print("=== FINAL SUMMARY MATRIX: MODEL CONSISTENCY (Mean ± Std Dev) ===")
print("="*60)

summary_rows = []
for clf_name in classifiers_config:
    aucs = results_history[clf_name]['roc_auc']
    accs = results_history[clf_name]['accuracy']
    precisions = results_history[clf_name]['precision']
    recalls = results_history[clf_name]['recall']
    f1s = results_history[clf_name]['f1_macro']
    fbetas = results_history[clf_name]['fbeta']

    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_precision, std_precision = np.mean(precisions), np.std(precisions)
    mean_recall, std_recall = np.mean(recalls), np.std(recalls)
    mean_f1, std_f1 = np.mean(f1s), np.std(f1s)
    mean_fbeta, std_fbeta = np.mean(fbetas), np.std(fbetas)

    individual_metrics = {
        'Model': [clf_name for _ in range(N_REPETITIONS)],
        'ROC-AUC': aucs,
        'Accuracy': accs,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Macro': f1s,
        'F-Beta': fbetas
    }
    df_summary = pd.DataFrame(individual_metrics)
    pprint.pprint(individual_metrics)

    os.makedirs("results/assessment/vIAbility/tabular", exist_ok=True)
    df_summary.to_csv("results/assessment/vIAbility/tabular/ml_models_individual.csv", index=False)

    summary_rows.append({
        'Model': clf_name,
        'ROC-AUC Summary': f"{mean_auc:.4f} ± {std_auc:.4f}",
        'Accuracy Summary': f"{mean_acc:.4f} ± {std_acc:.4f}",
        'Precision Summary': f"{mean_precision:.4f} ± {std_precision:.4f}",
        'Recall Summary': f"{mean_recall:.4f} ± {std_recall:.4f}",
        'F1-Macro Summary': f"{mean_f1:.4f} ± {std_f1:.4f}",
        'F-Beta Summary': f"{mean_fbeta:.4f} ± {std_fbeta:.4f}"
    })

df_summary = pd.DataFrame(summary_rows)
pprint.pprint(summary_rows)


df_summary.to_csv("results/assessment/vIAbility/tabular/ml_models_consistency_report.csv", index=False)