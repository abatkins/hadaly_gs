from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from MPINestedGridSearchCV import NestedGridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from get_variables import VariablesXandY
from sklearn.cross_validation import ShuffleSplit
import logging
from sklearn.externals import joblib
from os import path, remove, listdir, makedirs
from mpi4py import MPI
import pandas as pd
import argparse

rank = MPI.COMM_WORLD.Get_rank()
master = bool(rank == 0)

# Handles jobdir creation. This is where logs and output for the job go.
def create_jobdir(prod, jobname):
    if prod:
        base_dir = "../scr00"
    else:
        base_dir = ""

    output_dir = path.join(base_dir, 'output')
    job_dir = path.join(output_dir, jobname)

    try:
        makedirs(job_dir)
    except OSError as e:
        if e.errno != 17:
            raise  # this is not a "directory exists" error

    fileList = listdir(job_dir)
    if master and fileList:
        for fileName in fileList:
            file_path = path.join(job_dir, fileName)
            remove(file_path)

    return job_dir

def main(args):
    train_file = args.filename
    prod = args.prod
    gridsearch = args.gridsearch
    jobname = args.jobname
    cv_type = args.cv
    dump = args.dump

    log_filename = 'gridsearch.log'
    job_dir = create_jobdir(prod, jobname)
    log_path = path.join(job_dir, log_filename)
    if dump:
        pickle_path = path.join(job_dir, 'pickles')
        label_path = path.join(pickle_path, 'labels.pkl')
    else:
        pickle_path = None
        label_path = None
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s %(message)s')

    df_whole_data = pd.read_csv(train_file, sep=',', quotechar='"', encoding='utf-8')
    text = df_whole_data['text']

    variables_object = VariablesXandY(input_filename=df_whole_data)
    y_train = variables_object.get_y_matrix(labels_pickle_filename=label_path).todense()
    #x_train = variables_object.get_x(text, n_gram)


    #### This appears to be the correct way to combine these. Try this implementation.
    # Perform an IDF normalization on the output of HashingVectorizer
    n_gram = (1, 2)
    hash = HashingVectorizer(ngram_range=n_gram, stop_words='english', strip_accents="unicode"#,
                             #non_negative=True, norm=None,
                             #token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b" # tokens are character strings of 2 or more characters
    )
    vect = make_pipeline(hash, TfidfTransformer())
    x_train = vect.fit_transform(text)

    #rbm = BernoulliRBM(random_state=0, verbose=True)
    #svc = LinearSVC(class_weight="balanced")
    sgd = SGDClassifier(n_jobs=1, random_state=0)
    pipe = Pipeline(steps=[
        #('rbm', rbm),
        ('sgd', sgd)
        #('svc', svc)
    ])

    model_to_set = OneVsRestClassifier(pipe, n_jobs=1)

    # k folds, p paramaters, n options
    # number of model fits is equal to k*n^p
    # Ex: 3*2^4 = 48 for this case
    parameters = {
        'estimator__sgd__loss': ['squared_hinge'],#['hinge','squared_hinge','log','modified_huber','perceptron'], # squared_hinge same as linear svc
        'estimator__sgd__penalty': ['l2'], #['l2','l1','elasticnet']# l2 is same as linear svc
        'estimator__sgd__n_iter': [15, 20, 25, 30],
        'estimator__sgd__alpha': [0.00001],
        #'estimator__sgd__l1_ratio': [0.01, 0.15, 0.3, 0.5], # use with elasticnet
        #'estimator__sgd__learning_rate': ['constant, optimal, invscaling'],
        #'estimator__sgd__eta0': [0.0, 0.01, .10, 0.3], # used when learning rate is constant or invscaling
        #'estimator__sgd__power_t': [0.01, 0.2, 0.5, 0.75], # exponent used in invscaling
        #"estimator__rbm__batch_size": [5,10], #[5,10]
        #"estimator__rbm__learning_rate": [.06,.1],#[.001, .01, .06, .1],
        #"estimator__rbm__n_iter": [2,5],#[1,2,4,8,10],
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        #"estimator__svc__C": [1000, 10, 1, .01] #[.01, 1, 10, 100, 1000, 10000]
    }
    f1_scorer = make_scorer(f1_score, average='samples')

    # Handle CV method
    if cv_type == "shufflesplit":
        custom_cv = ShuffleSplit(len(y_train), n_iter=5, test_size=0.20, random_state=0)  # iters should be higher than inner
        new_size = int(len(y_train) * (1 - custom_cv.test_size))
        custom_inner_cv = lambda _x, _y: ShuffleSplit(new_size, n_iter=3, test_size=0.10, random_state=1)
    else:
        custom_cv = 5
        custom_inner_cv = 3

    # Handle Gridsearch (Nested, Normal, None)
    if gridsearch == "nested":
        model_tunning = NestedGridSearchCV(model_to_set,
                                           param_grid=parameters,
                                           scoring=f1_scorer,
                                           cv=custom_cv,
                                           inner_cv=custom_inner_cv,
                                           multi_output=True
                                           )
    elif gridsearch == "none":
        custom_cv = ShuffleSplit(len(y_train), n_iter=1, test_size=0.20, random_state=0)
        for train_set, test_set in custom_cv:
            x_test = x_train[test_set]
            y_test = y_train[test_set]
            x_train = x_train[train_set]
            y_train = y_train[train_set]

        model_tunning = model_to_set
    else: # normal gridsearch
        model_tunning = GridSearchCV(model_to_set, param_grid=parameters, scoring=f1_scorer, cv=custom_cv)

    if master:
        logging.info("Fitting model...")
        logging.info(vect)
        logging.info(model_tunning)
    model_tunning.fit(x_train, y_train)

    if dump:
        logging.info("Dumping model...")
        model_path = path.join(pickle_path, 'model.pkl')
        joblib.dump(model_tunning, model_path)

    if master:
        if gridsearch == "nested":
            for i, scores in enumerate(model_tunning.grid_scores_):
                csv_file = path.join(job_dir, 'grid-scores-%d.csv' % (i + 1))
                scores.to_csv(csv_file, index=False)

            print(model_tunning.best_params_)
            logging.info('cv used:' + str(custom_cv))
            logging.info("best params: " + str(model_tunning.best_params_))
        elif gridsearch == "none":
            y_pred = model_tunning.predict(x_test)
            score = f1_score(y_test, y_pred, average='samples')
            logging.info("f1_score: %s" % str(score))
        else: # normal gridsearch
            print(model_tunning.best_params_)
            logging.info('cv used:' + str(custom_cv))
            logging.info("best params: " + str(model_tunning.best_params_))

        logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hadaly GridsearchCV")
    parser.add_argument("--prod", help="Set environment to production", action="store_true")
    parser.add_argument("--dump", help="Dump fitted model to pickle file", action="store_true")
    parser.add_argument("-g", "--gridsearch", help="Select gridsearch type", choices=['nested', 'normal', 'none'], default="normal")
    parser.add_argument("-c", "--cv", help="Select CV method", type=str, choices=['shufflesplit', 'kfold'], default="kfold")
    parser.add_argument("-f", "--filename", help="filename", type=str, default="test.csv")
    parser.add_argument("-j", "--jobname", help="jobname (specifies output directory)", type=str, default="your_job")
    args = parser.parse_args()

    main(args)