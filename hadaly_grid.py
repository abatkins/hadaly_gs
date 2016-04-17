from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from MPINestedGridSearchCV import NestedGridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from get_variables import VariablesXandY
from sklearn.cross_validation import ShuffleSplit
import logging
#from sklearn.externals import joblib
from os import path, remove
from mpi4py import MPI

def main(prod, nested):
    rank = MPI.COMM_WORLD.Get_rank()
    LOG_FILENAME = 'logs/gridsearch.log'

    if rank == 0 and path.isfile(LOG_FILENAME):
        remove(LOG_FILENAME)

    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG, format='%(asctime)s %(message)s')
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
    svc = LinearSVC(class_weight="balanced")
    pipe = Pipeline(steps=[
        #('rbm', rbm),
        ('svc', svc)
    ])

    model_to_set = OneVsRestClassifier(pipe, n_jobs=1)

    # k folds, p paramaters, n options
    # number of model fits is equal to k*n^p
    # Ex: 3*2^4 = 48 for this case
    parameters = {
        #"estimator__rbm__batch_size": [5,10], #[5,10]
        #"estimator__rbm__learning_rate": [.06,.1],#[.001, .01, .06, .1],
        #"estimator__rbm__n_iter": [2,5],#[1,2,4,8,10],
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        "estimator__svc__C": [1000]
    }
    f1_scorer = make_scorer(f1_score, average='samples')

    #custom_cv = ShuffleSplit(len(y_train), n_iter=5, test_size=0.20, random_state=0) # iters should be higher
    custom_cv = 5
    #custom_inner_cv = 5
    custom_inner_cv=lambda _x, _y: ShuffleSplit(int(len(y_train)*.99), n_iter=3, test_size=0.01, random_state=1)

    if nested:
        model_tunning = NestedGridSearchCV(model_to_set,
                                           param_grid=parameters,
                                           scoring=f1_scorer,
                                           cv=custom_cv,
                                           inner_cv=custom_inner_cv,
                                           multi_output=True
        )
    else:
        model_tunning = GridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer, cv=custom_cv)

    logging.info("Fitting model...")
    model_tunning.fit(x_train, y_train)

    #output_path = path.join(base_dir,'output/output.pkl')

    #logging.info("Dumping model...")
    #joblib.dump(model_tunning, output_path)

    if rank == 0:
        for i, scores in enumerate(model_tunning.grid_scores_):
            csv_file = path.join(base_dir,'output/grid-scores-%d.csv' % (i + 1))
            scores.to_csv(csv_file, index=False)

        #print(model_tunning.best_score_)
        print(model_tunning.best_params_)

        #logging.info("best score: " + str(model_tunning.best_score_))
        logging.info('cv used:' + str(custom_cv))
        logging.info("best params: " + str(model_tunning.best_params_))
        #logging.info("best estimator: " + str(model_tunning.best_estimator_))

        logging.info("Done!")

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
