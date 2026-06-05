#Installing and Downloading the required packages
#!pip install git+https://github.com/facebookresearch/segment-anything.git
#!pip install opencv-python pycocotools matplotlib onnxruntime onnx
#pip install SemTorch
#!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ../models/sam_vit_h_4b8939.pth

# PATHS to things

PATH_DATASET='/data/mvillota/Datasets/Datasets/Processed/'
PATH_MODELS='models/'
TMP_PATH = 'tmp/'
OUTPUT_PATH = 'output/'
SEED = 42

SEGMENTATION_VALUES = {'ZP': 75,'ICM': 150,'TE': 255}
GRADING_VALUES = {'A': 0, 'B': 1, 'C': 2}

IMAGE_SIZE = (224, 224)

PARAM_GRID = {'RandomForest': {'n_estimators': [50, 100, 200],
                              'criterion': ['gini', 'entropy', 'log_loss'],
                              'max_depth': [None, 5, 10, 20, 30, 40, 50],
                              'random_state': [SEED], "class_weight": ["balanced", None],},
              'SVC':{'C': [0.1, 1, 10, 100],
                   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                   'degree': [2, 3, 4],
                   'decision_function_shape': ['ovo', 'ovr'],
                   'random_state': [SEED], "class_weight": ["balanced", None],},
              "GaussianNB": {
                  "var_smoothing": [1e-9, 1e-7, 1e-5],},
              'DecisionTree':{'criterion': ['gini', 'entropy', 'log_loss'],
                            'splitter': ['best', 'random'],
                            'max_depth': [None, 5, 10, 20, 30, 40, 50],
                            'random_state': [SEED], "class_weight": ["balanced", None],},
              'MLPClassifier':{'hidden_layer_sizes': [(5,2), (10,5), (20,10),(5,5,2), (10,5,2), (5,3,2)],
                             'activation': ['relu', 'tanh', 'logistic'],
                             'solver': ['sgd', 'adam'],
                             "early_stopping":     [True],
                             'alpha': [1e-5, 1e-4, 1e-3],
                             'random_state': [SEED]},
              'LogisticRegression':{'C': [0.01, 0.1, 1, 10, 100],
                                  'solver': ['lbfgs', 'liblinear', 'saga'],
                                  'random_state': [SEED], "class_weight": ["balanced", None]},
                'TabPFN': {'ignore_pretraining_limits':[True]},
            }





