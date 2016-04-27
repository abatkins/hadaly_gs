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
#from sklearn.externals import joblib
from os import path, remove, listdir, makedirs, getcwd
import datetime
from mpi4py import MPI
import pandas as pd

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
            file_path = path.join(output_dir, fileName)
            remove(file_path)

    return job_dir

def main(prod, nested, jobname):
    train_file = 'test.csv'
    n_gram = (1, 2)
    log_filename = 'gridsearch.log'

    job_dir = create_jobdir(prod, jobname)
    log_path = path.join(job_dir, log_filename)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, format='%(asctime)s %(message)s')

    df_whole_data = pd.read_csv(train_file, sep=',', quotechar='"', encoding='utf-8')
    variables_object = VariablesXandY(input_filename=df_whole_data)
    y_train = variables_object.get_y_matrix().todense()

    text = df_whole_data['text']
    #x_train = variables_object.get_x(text, n_gram)


    #### This appears to be the correct way to combine these. Try this implementation.
    # Perform an IDF normalization on the output of HashingVectorizer
    #hasher = HashingVectorizer(ngram_range=n_gram, stop_words='english', strip_accents="unicode", non_negative=True, norm=None, token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b") # tokens are character strings of 2 or more characters
    hasher = HashingVectorizer(ngram_range=n_gram, stop_words="english", strip_accents="unicode")
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    x_train = vectorizer.fit_transform(text)

    #x_train_counts = hash_vect_object.fit_transform(text)
    #x_train_tfidf = tfidf_transformer_object.fit_transform(x_train_counts)

    #rbm = BernoulliRBM(random_state=0, verbose=True)
    svc = LinearSVC(class_weight="balanced")
    #sgd = SGDClassifier(n_iter=15, warm_start=True, n_jobs=-1, random_state=0, fit_intercept=True)
    pipe = Pipeline(steps=[
        #('sgd', sgd),
        #('rbm', rbm),
        ('svc', svc)
    ])

    model_to_set = OneVsRestClassifier(pipe, n_jobs=1)

    # k folds, p paramaters, n options
    # number of model fits is equal to k*n^p
    # Ex: 3*2^4 = 48 for this case
    parameters = {
        #'estimator__sgd__loss': 'hinge',
        #'estimator__sgd__penalty': 'l2',
        #'estimator__sgd__n_iter': 50,
        #'estimator__sgd__alpha': 0.00001,
        #"estimator__rbm__batch_size": [5,10], #[5,10]
        #"estimator__rbm__learning_rate": [.06,.1],#[.001, .01, .06, .1],
        #"estimator__rbm__n_iter": [2,5],#[1,2,4,8,10],
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        #"estimator__rbm__n_components": [3,5], #[1,5,10,20,100,256]
        "estimator__svc__C": [1000, 10, 1, .01] #[.01, 1, 10, 100, 1000, 10000]
    }
    f1_scorer = make_scorer(f1_score, average='samples')

    #custom_cv = ShuffleSplit(len(y_train), n_iter=5, test_size=0.20, random_state=0) # iters should be higher
    custom_cv = 5
    custom_inner_cv = 3
    #custom_inner_cv=lambda _x, _y: ShuffleSplit(int(len(y_train)*.99), n_iter=3, test_size=0.01, random_state=1)

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

    if master:
        logging.info("Fitting model...")
        logging.info(model_tunning)
    model_tunning.fit(x_train, y_train)

    #output_path = path.join(base_dir,'output/output.pkl')

    #logging.info("Dumping model...")
    #joblib.dump(model_tunning, output_path)

    if master:
        if nested:
            for i, scores in enumerate(model_tunning.grid_scores_):
                csv_file = path.join(job_dir,'grid-scores-%d.csv' % (i + 1))
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
    prod, nested = (False, False)
    jobname = "your_job"
    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == "--prod":
            prod = True
        if args[i] == "--nested":
            nested = True
        if args[i] not in ["--prod", "--nested"]:
            jobname = args[i]

    main(prod, nested, jobname)
