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
import os.path

def main(arg):
    if arg == "--prod":
        logging.info("Env: production")
        base_dir = "../scr00"
    else:
        logging.info("Env: development")
        base_dir = ""
    LOG_FILENAME = 'logs/gridsearch.log'
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

    train_file = 'test.csv'
    label_file = os.path.join(base_dir,'output/labels.pkl')
    variables_object = VariablesXandY(input_filename=train_file)
    y_train = variables_object.get_y_matrix(labels_pickle_filename=label_file).todense()
    n_gram = (1,2)
    x_train = variables_object.get_x_matrix(n_gram)

    rbm = BernoulliRBM(random_state=0, verbose=True)
    svc = LinearSVC(C=1000, class_weight="auto")
    pipe = Pipeline(steps=[('rbm', rbm), ('svc', svc)])

    model_to_set = OneVsRestClassifier(pipe, n_jobs=1)

    parameters = {
        "estimator__rbm__batch_size": [5,10], #[5,10]
        "estimator__rbm__learning_rate": [.06,.1],#[.001, .01, .06, .1],
        "estimator__rbm__n_iter": [2,5],#[1,2,4,8,10],
        "estimator__rbm__n_components": [3,5]#[1,5,10,20,100,256]
    }

    f1_scorer = make_scorer(f1_score, average='samples')

    model_tunning = NestedGridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer, multi_output=True)
    #model_tunning = GridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer)
    model_tunning.fit(x_train, y_train)

    output_path = os.path.join(base_dir,'output/output.pkl')
    joblib.dump(model_tunning, output_path)

    logging.info("best score: " + str(model_tunning.best_score_))
    logging.info("best params: " + str(model_tunning.best_params_))

    logging.info("best estimator: " + str(model_tunning.best_estimator_))
    logging.info("grid scores: " + str(model_tunning.grid_scores_))

    print(model_tunning.best_score_)
    print(model_tunning.best_params_)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 1:
        arg = sys.argv[1]
    else:
        arg = "--dev"
    main(arg)