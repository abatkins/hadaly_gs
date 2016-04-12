from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from MPINestedGridSearchCV import NestedGridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from get_variables import VariablesXandY
import logging
from sklearn.externals import joblib
from os import path, remove

def main(prod,nested):
    LOG_FILENAME = 'logs/gridsearch.log'

    if path.isfile(LOG_FILENAME):
        remove(LOG_FILENAME)
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

    if prod:
        logging.info("Env: production")
        base_dir = "../scr00"
    else:
        logging.info("Env: development")
        base_dir = ""

    train_file = 'test.csv'
    label_file = path.join(base_dir,'output/labels.pkl')
    variables_object = VariablesXandY(input_filename=train_file)
    y_train = variables_object.get_y_matrix(labels_pickle_filename=label_file).todense()
    n_gram = (1,2)
    x_train = variables_object.get_x_matrix(n_gram)

    rbm = BernoulliRBM(random_state=0, verbose=True)
    svc = LinearSVC(class_weight="auto")
    pipe = Pipeline(steps=[
        ('rbm', rbm),
        ('svc', svc)
    ])

    model_to_set = OneVsRestClassifier(pipe, n_jobs=1)

    parameters = {
        "estimator__rbm__batch_size": [5,10], #[5,10]
        "estimator__rbm__learning_rate": [.06,.1],#[.001, .01, .06, .1],
        "estimator__rbm__n_iter": [2,5],#[1,2,4,8,10],
        "estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        "estimator__svc__C": [1000]
    }

    f1_scorer = make_scorer(f1_score, average='samples')

    if nested:
        model_tunning = NestedGridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer, multi_output=True)
    else:
        model_tunning = GridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer)

    logging.info("Fitting model...")
    model_tunning.fit(x_train, y_train)

    output_path = path.join(base_dir,'output/output.pkl')

    logging.info("Dumping model...")
    joblib.dump(model_tunning, output_path)

    logging.info("best score: " + str(model_tunning.best_score_))
    logging.info("best params: " + str(model_tunning.best_params_))

    logging.info("best estimator: " + str(model_tunning.best_estimator_))
    logging.info("grid scores: " + str(model_tunning.grid_scores_))

    print(model_tunning.best_score_)
    print(model_tunning.best_params_)

if __name__ == "__main__":
    import sys
    prod, nested = (False,False)
    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == "--prod":
            prod = True
        if args[i] == "--nested":
            nested = True
    main(prod, nested)