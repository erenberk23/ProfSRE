 
from math import ceil
# from empulse.metrics import empcs
# from empulse.metrics import empcs_score
# from empulse.metrics import mpcs_score
import tempfile 
from PIL import Image
import numpy as np
# from EMP.metrics import empCreditScoring
import pandas as pd
import re
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import _tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import rpy2
import os
import sys
import ast
sys.path.append('../test')
sys.path.append('../lib')
sys.path.append('../proflogit')
from rpy2.robjects import Formula
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import FloatVector, IntVector
from rpy2.robjects import r as R
import ctypes
from sklearn.metrics import confusion_matrix
from rpy2.robjects import pandas2ri, packages,numpy2ri
pandas2ri.activate()
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import multiprocessing
r = robjects.r
#import importlib
#import glmnet_python
#from glmnet_python import glmnet,glmnetPlot,glmnetPrint,glmnetCoef,glmnetPredict,cvglmnet,cvglmnetCoef,cvglmnetPlot,cvglmnetPredict
from sklearn.metrics import roc_auc_score, log_loss
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
utils = rpackages.importr('utils')
# Gerekli R paketleri
#required_packages = ['glmnet' ,'oemEren','doParallel','rpart','gbm']
required_packages = ['glmnet','oem' ,'doParallel','rpart','gbm']
# Eksik paketleri yükleme

utils.chooseCRANmirror(ind=1)  # CRAN aynasını seçin (1 = global)
packages_to_install = [pkg for pkg in required_packages if not rpackages.isinstalled(pkg)]

if len(packages_to_install) > 0:
    utils.install_packages(StrVector(packages_to_install))

rpart = rpackages.importr('rpart')
glmnet = rpackages.importr('glmnet')
robjects.r('if (!requireNamespace("oem", quietly = TRUE)) install.packages("oem")')
#robjects.r('if (!requireNamespace("oemEren", quietly = TRUE)) install.packages("oemEren")')
# Load the oem package
robjects.r('library(oem)')
#robjects.r('library(oemEren)')
doParallel = rpackages.importr('doParallel')
robjects.r('''library(mgcv)''')
robjects.r('library(rpart)')
robjects.r('library(glmnet)')
robjects.r('library(gbm)')
robjects.r('''
packagelist <- c("glmnet", "stringr", "pdp", "mgcv", "caret", "pROC", "doParallel", "rpart", "SGL", "lsgl", "oem", "randomForest", "sglfast", "caTools", "EMP", "sourceCpp", "evtree", "Rcpp","gbm")

# Eksik R paketlerini yükleme
new.packages <- packagelist[!(packagelist %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# R paketlerini yükleme
lapply(packagelist, require, character.only = TRUE)
''')
 

robjects.r('library(doParallel)')
base = rpackages.importr('base')
utils = rpackages.importr('utils')
emp = rpackages.importr('EMP')
pandas2ri.activate()
wd = "C:/Users/test1/OneDrive - Roof Stacks Yazılım Anonim Şirketi/Masaüstü/Cell2Cell"

os.chdir(wd)
if not rpackages.isinstalled('rpart'):
    rpackages.install_packages('rpart')
    rpart = rpackages.importr('rpart')
if not rpackages.isinstalled('glmnet'):
    rpackages.install_packages('glmnet')
    glmnet = rpackages.importr('glmnet')
if not rpackages.isinstalled('doParallel'):
    rpackages.install_packages('doParallel')
    doParallel = rpackages.importr('doParallel')
r_predict_gam = robjects.r['predict.gam']



# Fonksiyon: Kuralları normalize eder
def normalize_logic(rule):
    match = re.match(r"(\d)\s*if\s*(.+?)\s*else\s*(\d)", rule)
    if not match:
        return rule.strip()
    
    then_val, condition, else_val = match.groups()
    conditions = [c.strip() for c in condition.split("and")]
    conditions_sorted = sorted(conditions)
    normalized_condition = ' and '.join(conditions_sorted)
    
    if then_val == '0' and else_val == '1':
        return f"NEGATED:({normalized_condition})"
    elif then_val == '1' and else_val == '0':
        return f"NORMAL:({normalized_condition})"
    else:
        return rule.strip()
def tree(traindata, tree_algorithm="CART", pruning=True, output_pred=True,
         minbucketsize="AUTO", CART_cost_matrix=[[0, 1], [1, 0]], CART_cost_vector=None, maxdepth=30):

    formula = "dependent ~ " + " + ".join(traindata.columns[:-1])
    X = traindata.iloc[:, :-1]
    y = traindata.iloc[:, -1]
    depvalues = sorted(y.unique())
    maxdepth = len(np.unique(depvalues))
    n = len(traindata)

    # Pruning options
    # CART
    cp = 0.1

    if tree_algorithm == "CART":
        if CART_cost_vector is None:
            caseweights = [CART_cost_matrix[0][1] - CART_cost_matrix[0][0] if yi == "cl0"
                           else CART_cost_matrix[1][0] - CART_cost_matrix[1][1] for yi in y]
            caseweights = caseweights / np.max(caseweights)
        else:
            caseweights = CART_cost_vector / max(CART_cost_vector)

        if minbucketsize != "AUTO":
            dt = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth, min_samples_leaf=minbucketsize)
        else:
            dt = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth)

        dt.fit(X, y, sample_weight=caseweights)
        splits_info = dt.tree_
        tree_to_code(dt,dt.feature_names_in_)
        print(splits_info.__getstate__()['nodes'][:5])
        fig = plt.figure(figsize=(15, 12))
        plot_decision_tree(dt,dt.feature_names_in_,dt.classes_)
        plt.show()
        if pruning == True:
            # TODO: implement pruning
            pass

        pred_prob = dt.predict_proba(X)
        pred_class = dt.predict(X)

    pred_class2 = pred_class.astype(str)
    for i, val in enumerate(depvalues):
        pred_class2[(pred_class == i)] = str(val)

    pred_class = pred_class2
    del pred_class2

    ind = (y != pred_class)
    error_rate = sum(ind) / n

    if output_pred == True:
        pred_all = pd.DataFrame({"dependent": y, "pred": pred_class, **{f"prob_{i}": pred_prob[:,i] for i in range(len(depvalues))}})
    else:
        pred_all = None

    ans = {"pred": pred_all, "formula": formula, "tree": dt, "tree_algorithm": tree_algorithm,
           "pruning": pruning, "error_rate": error_rate, "CART.cost.matrix": CART_cost_matrix}
    return ans
def tree2(traindata, tree_algorithm="CART", pruning=True, output_pred=True,
         minbucketsize="AUTO", CART_cost_matrix=[[0,1],[1,0]], CART_cost_vector=None, maxdepth=30):

    formula =Formula("dependent ~ " + " + ".join(traindata.columns[:-1]))

    #X = traindata.iloc[:, :-1]
    y = traindata.iloc[:, -1]
    vardep = traindata.iloc[:, -1]
    maxdepth = len(vardep.unique())
    tmp = vardep.unique()
    depvalues = np.sort(tmp)

    ##depvalues = sorted(y.unique())
    maxdepth = len(np.unique(depvalues))
    n = len(traindata)

    # Pruning options
    # CART
    cp = 0.1

    if tree_algorithm == "CART":
        if CART_cost_vector is None:
            caseweights = [CART_cost_matrix[0][1] - CART_cost_matrix[0][0] if yi == "cl0"
                           else CART_cost_matrix[1][0] - CART_cost_matrix[1][1] for yi in y]
            caseweights = caseweights / np.max(caseweights)
        else:
            caseweights = CART_cost_vector / max(CART_cost_vector)
    pandas2ri.activate()


    if minbucketsize != "AUTO":
      fit = rpart.rpart(formula, data=traindata, method="class", control=rpart.control(maxdepth=maxdepth, minbucket=minbucketsize), weights=caseweights,x=True,y=True)
    else:
      fit = rpart.rpart(formula, data=traindata, method="class", weights=caseweights, maxdepth=maxdepth)


    if pruning is True:
      fit = rpart.prune(fit, cp=cp)


    pred_prob = rpart.predict_rpart(fit, newdata=traindata, type="prob")
    pred_class = pd.DataFrame(rpart.predict_rpart(fit, newdata=traindata, type="class")).astype(int)

    pred_class2 = pred_class.astype(str)
    for i in range(1, len(depvalues) + 1):
      pred_class2[pred_class == i] = str(depvalues[i - 1])

    pred_class = pred_class2.values.flatten()
    del pred_class2

    ind = np.where(vardep != pred_class)[0]
    error_rate = len(ind) / n

    if output_pred is True:
      pred_all = pd.concat([vardep.reset_index(drop=True), pd.Series(pred_class, name="pred"), pd.DataFrame(pred_prob)], axis=1)
      pred_all.columns = ["dependent"] + ["pred"] + ["prob_" + str(i) for i in range(1, pred_prob.shape[1] + 1)]
    else:
      pred_all = None

    ans = {'pred': pred_all, 'formula': formula, 'tree': fit, 'tree_algorithm': tree_algorithm, 'pruning': pruning, 'error_rate': error_rate, 'CART_cost_matrix': CART_cost_matrix}

    return ans

def treeToProfTree(traindata, tree_algorithm="ProfTree", pruning=True, output_pred=True,
         minbucketsize="AUTO", maxdepth=10, Lambda=0.2, seed=2020):
    formula =Formula("dependent ~ " + " + ".join(traindata.columns[:-1]))
    if tree_algorithm == "ProfTree":
        # Python'dan R'a veri setini geçir
        r_traindata = pandas2ri.py2rpy(traindata)
        robjects.globalenv['r_traindata'] = r_traindata;

        robjects.r("r_traindata$dependent <- as.factor(r_traindata$dependent)")
        robjects.r(f'''
        set.seed(2020)

        ProfTree<- proftree(dependent ~ ., data = r_traindata, control = proftree.control(lambda =0.01,seed = 2020,verbose=TRUE,p0=0.55,p1=0.1,ROI=0.2644, ,minbucket=10))
        ''')

        fit = robjects.globalenv['ProfTree']

        # Tahminleme yap ve kuralları geri al
        predict_proftree = robjects.r('predict(ProfTree, newdata=r_traindata, type="response",observed = r_traindata$choice)')
        pred_class = np.array(predict_proftree)



        # Hata oranını hesapla
        y_true = traindata.iloc[:, -1]
        error_rate = np.mean(y_true != pred_class)

        # Sonuçları döndür
        pred_all = pd.DataFrame({'true': y_true, 'pred': pred_class})
        ans = {'pred': pred_all, 'formula': formula, 'tree': fit, 'tree_algorithm': tree_algorithm, 'pruning': pruning, 'error_rate': error_rate}
        return ans
    else:
        print("Only ProfTree is supported in this version.")
def draw_tree():
    fit=model['rules.trees'][9]['tree']
    globalenv['my_tree'] = fit
    robjects.r('''
               library(partykit)
               png("tree_output.png", width=800, height=600)
               plot(my_tree)
               text(my_tree, use.n=TRUE)
               dev.off()
               ''')

    from IPython.display import Image, display
    display(Image("tree_output.png"))
    
def build_tree(traindata, tree_algorithm="CART", pruning=True, output_pred=True,
               minbucketsize="AUTO", CART_cost_matrix=np.array([[0, 1], [1, 0]]),
               CART_cost_vector=None, maxdepth=30):

    # Convert traindata to a Pandas dataframe
    traindata = pd.DataFrame(traindata)

    # Extract dependent variable and formula
    dependent = traindata.columns[-1]
    formula = f"{dependent} ~ {' + '.join(traindata.columns[:-1])}"

    # Determine number of unique values of dependent variable
    vardep = traindata[dependent].unique()
    maxdepth = len(vardep)
    depvalues = np.sort(vardep)
    n = len(traindata)

    # Set pruning options
    cp = 0.1
    if tree_algorithm == "CART":

        # Create cost vector
        if CART_cost_vector is None:
            caseweights = np.ones(n)
            caseweights[traindata[dependent]=="cl0"] = CART_cost_matrix[0,1]-CART_cost_matrix[0,0]
            caseweights[traindata[dependent]=="cl1"] = CART_cost_matrix[1,0]-CART_cost_matrix[1,1]
            caseweights = caseweights / max(caseweights)
        else:
            caseweights = CART_cost_vector / max(CART_cost_vector)
        fit <- rpart(formula,data=traindata, method="class",control=rpart.control(maxdepth=maxdepth,minbucket=minbucketsize),weights=t(caseweights))

        # Make predictions
        pred_prob = fit.predict_proba(traindata.iloc[:,:-1])
        pred_class = fit.predict(traindata.iloc[:,:-1])

    # Convert predicted class to original values
    pred_class2 = np.array(pred_class, dtype="str")
    for i in range(len(depvalues)):
        pred_class2[pred_class==i+1] = str(depvalues[i])
    pred_class = pred_class2
    del pred_class2

    # Calculate error rate
    ind = np.array(traindata[dependent] != pred_class)
    error_rate = np.sum(ind) / n

    # Create prediction output if specified
    if output_pred:
        pred_all = pd.concat([traindata[dependent], pd.Series(pred_class), pd.DataFrame(pred_prob)], axis=1)
        pred_all.columns = ["dependent", "pred"] + [f"prob_{depvalues[i]}" for i in range(len(depvalues))]
    else:
        pred_all = None

    # Return output as a dictionary
    output =  {"pred": pred_all, "formula": formula, "tree": fit, "tree_algorithm": tree_algorithm,
            "pruning": pruning, "error_rate": error_rate, "CART.cost.matrix": CART_cost_matrix}

    return output


def fit_tree(traindata, tree_algorithm="CART", pruning=True, output_pred=True,
             minbucketsize="AUTO", CART_cost_matrix=np.array([[0,1],[1,0]]),
             CART_cost_vector=None, maxdepth=30):

    formula = traindata.columns[0] + ' ~ ' + ' + '.join(traindata.columns[1:])
    vardep = traindata.iloc[:,0]
    maxdepth = len(np.unique(vardep))
    tmp = np.unique(vardep)
    depvalues = tmp[np.argsort(tmp)]
    n = len(traindata)

    # pruning options
    # CART
    cp = 0.1

    if tree_algorithm == "CART":
        # create cost vector
        if CART_cost_vector is None:
            caseweights = np.ones(traindata.shape[0])
            caseweights[traindata.iloc[:,-1] == "cl0"] = CART_cost_matrix[0,1] - CART_cost_matrix[0,0]
            caseweights[traindata.iloc[:,-1] == "cl1"] = CART_cost_matrix[1,0] - CART_cost_matrix[1,1]
            caseweights = caseweights / np.max(caseweights)
        else:
            caseweights = CART_cost_vector / np.max(CART_cost_vector)

        if minbucketsize != "AUTO":
            fit = DecisionTreeClassifier(criterion="gini", max_depth=maxdepth,
                                          min_samples_split=minbucketsize,
                                          class_weight="balanced")
        else:
            fit = DecisionTreeClassifier(criterion="gini", max_depth=maxdepth,
                                          class_weight="balanced")

        fit = fit.fit(traindata.iloc[:,1:], vardep, sample_weight=caseweights)

        if pruning:
            fit = fit.prune(cp=cp)

        pred_prob = fit.predict_proba(traindata.iloc[:,1:])
        pred_class = fit.predict(traindata.iloc[:,1:])

    pred_class2 = pred_class.astype(str)
    for i in range(len(depvalues)):
        pred_class2[pred_class==i+1] = depvalues[i]

    pred_class = pred_class2
    del pred_class2

    ind = (vardep != pred_class)
    error_rate = np.sum(ind) / n

    if output_pred:
        pred_all = pd.concat([vardep.reset_index(drop=True),
                              pd.Series(pred_class),
                              pd.DataFrame(pred_prob)], axis=1)
        pred_all.columns = ["dependent", "pred", "prob0", "prob1"]
    else:
        pred_all = None

    return {"pred": pred_all, "formula": formula, "tree": fit, "tree_algorithm": tree_algorithm,
            "pruning": pruning, "error_rate": error_rate, "CART.cost.matrix": CART_cost_matrix}

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))


    recurse(0, 1)

def sorensen_dice_index2(set1, set2):
    set1 = np.array(set1, dtype=bool)
    set2 = np.array(set2, dtype=bool)
    intersection = np.sum(set1 & set2)
    denominator = np.sum(set1) + np.sum(set2)
    return (2 * intersection) / denominator if denominator != 0 else 0

# Sorensen-Dice Stabilite Hesaplama
def sorensen_dice_index(set1, set2):
    intersection = np.sum(set1 & set2)
    denominator = np.sum(set1) + np.sum(set2)
    return (2 * intersection) / denominator if denominator != 0 else 0

def plot_decision_tree(model, feature_names, class_names):
    # plot_tree function contains a list of all nodes and leaves of the Decision tree
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)

    # I return the tree for the next part
    return tree
def tree(traindata, tree_algorithm="CART", pruning=True, output_pred=True,
         minbucketsize="AUTO", CART_cost_matrix=[[0, 1], [1, 0]], CART_cost_vector=None, maxdepth=30):

    formula = Formula("dependent ~ " + " + ".join(traindata.columns[:-1]))
    X = traindata.iloc[:, :-1]
    y = traindata.iloc[:, -1]
    depvalues = sorted(y.unique())
    maxdepth = len(np.unique(depvalues))
    n = len(traindata)

    # Pruning options
    # CART
    cp = 0.1

    if tree_algorithm == "CART":
        if CART_cost_vector is None:
            caseweights = [CART_cost_matrix[0][1] - CART_cost_matrix[0][0] if yi == "cl0"
                           else CART_cost_matrix[1][0] - CART_cost_matrix[1][1] for yi in y]
            caseweights = caseweights / np.max(caseweights)
        else:
            caseweights = CART_cost_vector / max(CART_cost_vector)

        if minbucketsize != "AUTO":
            dt = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth, min_samples_leaf=minbucketsize)
        else:
            dt = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth)

        dt.fit(X, y, sample_weight=caseweights)
        splits_info = dt.tree_
        tree_to_code(dt,dt.feature_names_in_)
        print(splits_info.__getstate__()['nodes'][:5])
        fig = plt.figure(figsize=(15, 12))
        plot_decision_tree(dt,dt.feature_names_in_,dt.classes_)
        plt.show()
        if pruning == True:
            # TODO: implement pruning
            pass

        pred_prob = dt.predict_proba(X)
        pred_class = dt.predict(X)

    pred_class2 = pred_class.astype(str)
    for i, val in enumerate(depvalues):
        pred_class2[(pred_class == i)] = str(val)

    pred_class = pred_class2
    del pred_class2

    ind = (y != pred_class)
    error_rate = sum(ind) / n

    if output_pred == True:
        pred_all = pd.DataFrame({"dependent": y, "pred": pred_class, **{f"prob_{i}": pred_prob[:,i] for i in range(len(depvalues))}})
    else:
        pred_all = None

    ans = {"pred": pred_all, "formula": formula, "tree": dt, "tree_algorithm": tree_algorithm,
           "pruning": pruning, "error_rate": error_rate, "CART.cost.matrix": CART_cost_matrix}
    return ans

def SRE_termcreator(traindata, valdata=None, rules_nrtrees=10, maxdepth=10, splines_df="AUTO", splinetype=["tp", "cr", "cs", "ps", "ts"]):

    if valdata==None:
        valdata = traindata

    vardep = traindata["dependent"]
    tmp = vardep.unique()
    depvalues = sorted(tmp)
    spline_type = ["tp", "cr", "cs", "ps", "ts"]

    n = len(traindata)
    p = len(traindata.columns) - 1
    nclasses = len(vardep.unique())

    member_trees = []
    rules = []

    member_pred = []
    add_rules = True
    add_splines = True
    add_lterms = True
    use_ProfTree = False
    add_stability= False
    if(use_ProfTree):
        robjects.r('''
extract_proftree_rules_combined <- function(node, data, condition = NULL, rules_list = list()) {
    # Eğer node bir yaprak düğümse
    if (is.null(node$split)) {
        # Yaprak düğümün tahmin sınıfını al
        node_id <- node$id
        if (node_id %in% data$`(fitted)`) {
            class_counts <- table(data$`(response)`[data$`(fitted)` == node_id])
            predicted_class <- ifelse(class_counts["cl1"] > class_counts["cl0"], 1, 0)
        } else {
            predicted_class <- 0  # Varsayılan
        }

        # Koşulları birleştirerek Python if-else formatında kural oluştur
        rule_text <- paste(predicted_class, "if",
                           paste(sapply(condition, function(cond) {
                               parts <- strsplit(cond, " ")[[1]]
                               paste0("(indata['", parts[1], "'] ", parts[2], " ", parts[3], ")")
                           }), collapse = " and "),
                           "else", 1 - predicted_class)
        rules_list <- c(rules_list, rule_text)
        return(rules_list)
    } else {
        # split_var'ı modelin terms özniteliğinden al
        split_var <- attr(terms(ProfTree), "term.labels")[node$split$varid]
        split_val <- node$split$breaks

        # Sol dal için koşul oluştur
        left_condition <- c(condition, paste0(split_var, " < ", split_val))
        rules_list <- extract_proftree_rules_combined(node$kids[[1]], data, left_condition, rules_list)

        # Sağ dal için koşul oluştur
        right_condition <- c(condition, paste0(split_var, " >= ", split_val))
        rules_list <- extract_proftree_rules_combined(node$kids[[2]], data, right_condition, rules_list)

        return(rules_list)
    }
}
    ''')



    if add_rules:
        for m in range(rules_nrtrees):
            print(f"training tree {m + 1}")
            bootstrap = np.random.choice(range(1, n),n, replace=True,)
            traindata_m = traindata.iloc[bootstrap]
            if(use_ProfTree) :
                member = treeToProfTree(traindata_m, tree_algorithm="ProfTree", pruning=True, output_pred=True, minbucketsize="AUTO", maxdepth=30, Lambda=0.1, seed=2020)
            else:
                member = tree2(traindata_m,tree_algorithm="CART",pruning=False,output_pred=False,maxdepth=10)
                tmptree =  predict_tree2(member,traindata)
            member_trees.append(member)

        ts = 0
        for m in range(rules_nrtrees):
            print(f"extracting rules for tree {m + 1}")
            if(use_ProfTree):
                robjects.globalenv['ProfTree']=member_trees[m]['tree']
                rules_list=robjects.r('''rules_list <- extract_proftree_rules_combined(ProfTree$node, ProfTree$fitted)''')
                rules_m=list(rules_list)
                rules_m = [str(rule) for rule in rules_m]
                rules_m = [ast.literal_eval(rule)[0] for rule in rules_m]
            else:
                rules_m = extract_rpart_rules(member_trees[m],type="binary")
                #print(rules)
            rules.append(rules_m)
            if "NONE" not in rules_m:
                ts += len(rules_m)

        n = len(valdata)
        member_pred = pd.DataFrame(np.zeros((n, ts)))
        ts = 0
        cnt = 0
        for m in range(rules_nrtrees):
            if "NONE" not in rules[m]:
                pred_prob_m = predict_rpart_rules(rules[m], valdata,predvarprefix="rule")
                nrr = pred_prob_m.shape[1]
                cnt += 1
                member_pred.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
                new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
                current_column_names = member_pred.columns.tolist()
                current_column_names[ts:ts+nrr] = new_column_names
                member_pred.columns = current_column_names
                ts += nrr
        print(f"extracting bitti")
        rulepreds = member_pred
        rules_support = rulepreds.mean()
        print("rules_support")
        print(rules_support)
        rules_scale = np.sqrt(rulepreds.mean() * (1 - rulepreds.mean()))
        ##rules_array = np.array(rules, dtype=object).reshape(-1, 1)
        ##rules_rulemaptable = pd.DataFrame(rules_array, columns=['V1'])
        flat_list = [item for sublist in rules for item in sublist]

        rules_rulemaptable = pd.DataFrame(flat_list, columns=['V1'])
        rules_rulemaptable = rules_rulemaptable[rules_rulemaptable != "NONE"]
        rules_rulemaptable = rules_rulemaptable[~rules_rulemaptable['V1'].isin(['N', 'O','E'])]
        rules_rulemaptable = rules_rulemaptable.reset_index(drop=True)
        colnames_df = pd.DataFrame(rulepreds.columns,columns=['V2'])
        rules_rulemaptable = pd.concat([rules_rulemaptable, colnames_df], axis=1)
        print("rules_rulemaptable")
        print(rules_rulemaptable)
        rules_support.index = rules_rulemaptable.index
        rules_rulemaptable['support'] = rules_support
        rules_scale.index = rules_rulemaptable.index
        rules_rulemaptable['scale'] = rules_scale
        uniquerules = rules_rulemaptable.drop_duplicates('V1')['V2']
        rulepreds = rulepreds[uniquerules]
        rules_rulemaptable.columns = ["rule", "term", "support", "scale"]

    else:
        rules = None
        rules_rulemaptable =None
    if add_stability:
        print("adding stability index")   
 
        # stabilities=[]
        # member_pred = member_pred.round().astype(int)  # 0 ve 1 değerlerine yuvarlama
        # stabilities = [
        #     sum(sorensen_dice_index(member_pred[f'rule{str(j+1).zfill(5)}'], member_pred[f'rule{str(i+1).zfill(5)}'])  
        #         for i in range((member_pred.columns.size)) if i != j) 
        #     for j in range((member_pred.columns.size))]
        n_rules = member_pred.shape[1]
        # Her bir rule için stability değeri (normalize edilmiş)
        stabilities = [
    np.mean([
        sorensen_dice_index2(member_pred.iloc[:, j], member_pred.iloc[:, i])
        for i in range(n_rules) if i != j
    ])
    for j in range(n_rules)
]
        
        df_stabilities = pd.DataFrame(stabilities, columns=['V1'])
        colnames_df_stabilities =pd.DataFrame([f"stability{str(i+1).zfill(5)}" for i in range(len(member_pred.columns))],columns=['term'])
        df_stabilities = pd.concat([df_stabilities, colnames_df_stabilities], axis=1)
        df_stabilities.columns = ["sindex", "term"]

    if add_splines:
        threshold = 20
        print("training splines")
        traindata_s = pd.concat([traindata.loc[:, traindata.apply(lambda col: len(col.unique())) > threshold], traindata['dependent']], axis=1)
        traindata_s.rename(columns={traindata_s.columns[-1]: "dependent"}, inplace=True)
        # to make sure we only estimate splines for variables where it makes sense
        splines = gamsplines(traindata_s, df=splines_df, gamtype="gam", threshold=threshold, splinetype=spline_type)
        splinepreds = predict_gamsplines(splines,valdata)
        splines_stddevs = np.zeros((splinepreds.shape[1], 1))
        for j in range(splinepreds.shape[1]):
            stddev = splinepreds.iloc[:, j].std()
            splines_stddevs[j] = 1 if stddev == 0 else stddev
        splinepreds_array = splinepreds.values
        splines_stddevs = splines_stddevs.flatten()
        splines_valdata = 0.4 * (splinepreds_array / splines_stddevs)
        splines_valdata = pd.DataFrame(splines_valdata, columns=splinepreds.columns)
    else:
        splines = None
        splines_stddevs = None

    if add_lterms:
        print("adding lterms")
        # Standart sapmaları hesapla
        lterms_stddevs = np.array([traindata.iloc[:, j].std() if traindata.iloc[:, j].std() != 0 else 1 for j in range(p)])

        # valdata'yı normalize et ve 0.4 ile çarp
        lterms_valdata = 0.4 * (valdata.iloc[:, :p].values / lterms_stddevs)
        ##lterms_valdata = valdata.iloc[:, :p].values
        lterms_valdata = pd.DataFrame(lterms_valdata, columns=traindata.columns[:p])
        # Sütun isimlerini al
        lterms = traindata.columns[:p]
        print("lterms added")
    else:
        lterms_stddevs = None
        lterms = None

    # Koşullara göre veri setlerini birleştir
    if add_lterms and add_rules and add_splines:
       termselmatrix = pd.concat([rulepreds, splines_valdata, lterms_valdata, valdata.iloc[:, -1]], axis=1)
    elif add_lterms and add_rules:
        termselmatrix = pd.concat([rulepreds, lterms_valdata, valdata.iloc[:, -1]], axis=1)
    elif add_rules:
        termselmatrix = pd.concat([rulepreds, valdata.iloc[:, [-1]]], axis=1)
    else:
        print("Rule ensemble without rules doesn't make sense. Computation halted")
        exit()
    ans = {'type': 'classification',
       'rules.rulemaptable': rules_rulemaptable,
       'rules.trees': member_trees,
       'splines': splines,
       'splines.stddevs': splines_stddevs,
       'lterms': lterms,
       'lterms.stddevs': lterms_stddevs,
       'rules': rules,
       'rules.nrtrees': rules_nrtrees,
       'classes': depvalues,
       'termselmatrix': termselmatrix,
       'terms.rules': rulepreds,
       'terms.splines': splines_valdata,
       'terms.linear': lterms_valdata,
       'stability':df_stabilities}

    ans = dict(ans)
    ans['type'] = "SREterms"
    return ans

def SRE(traindata, termmodel=False, valdata=None, rules_nrtrees=10, regular_type=["stepwisereg","lasso","ridge","elasticnet","SSVS","caret.optimized","SGL"],
        regular_alpha=0.5, add_rules=True, add_splines=True, add_lterms=True,
        maxdepth=10, splines_df='AUTO', parallel=False, nr_cores=1, cv_metric=['AUC','-LL'],
        sgl_alpha=0.95,maxit=1000,sgl_gamma=0.8,nlambda=20,sgl_lambdaselect=["sparsestInsignDiffFromBest","best"],sgl_reduce_ruleset=False,sgl_grouped_ruleset=False,cv_folds=1, standardize=False, splinetype=["tp", "cr", "cs", "ps", "ts"]):

    if valdata==None:
        valdata = traindata

    if regular_type == "lasso2":
        regular_alpha = 1
    if regular_type == "ridge":
        regular_alpha = 0


    vardep = traindata["dependent"]
    tmp = vardep.unique()
    depvalues = sorted(tmp)
    spline_type = ["tp", "cr", "cs", "ps", "ts"]

    n = len(traindata)
    p = len(traindata.columns) - 1
    nclasses = len(vardep.unique())

    member_trees = []
    rules = []
    add_stability = True
    member_pred = []
    lambdaseloverview = None
    if not isinstance(termmodel, dict):
        if add_rules:
            for m in range(rules_nrtrees):
                print(f"training tree {m + 1}")
                bootstrap = np.random.choice(range(1, n),n, replace=True,)
                traindata_m = traindata.iloc[bootstrap]
                member = tree2(traindata_m,tree_algorithm="CART",pruning=False,output_pred=False,maxdepth=10)
                tmptree =  predict_tree2(member,traindata)
                #pred_prob_m = tmptree.iloc[:, 2:]
                member_trees.append(member)

            ts = 0
            for m in range(rules_nrtrees):
                print(f"extracting rules for tree {m + 1}")
                rules_m = extract_rpart_rules(member_trees[m],type="binary")
                rules.append(rules_m)
                if "NONE" not in rules_m:
                    ts += len(rules_m)

            n = len(valdata)
            member_pred = pd.DataFrame(np.zeros((n, ts)))
            ts = 0
            cnt = 0
            for m in range(rules_nrtrees):
                if "NONE" not in rules[m]:
                    pred_prob_m = predict_rpart_rules(rules[m], valdata,predvarprefix="rule")
                    nrr = pred_prob_m.shape[1]
                    cnt += 1
                    member_pred.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
                    new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
                    current_column_names = member_pred.columns.tolist()
                    current_column_names[ts:ts+nrr] = new_column_names
                    member_pred.columns = current_column_names
                    ts += nrr

            rulepreds = member_pred
            rules_support = rulepreds.mean()
            rules_scale = np.sqrt(rulepreds.mean() * (1 - rulepreds.mean()))
            rules_array = np.array(rules).reshape(-1, 1)
            rules_rulemaptable = pd.DataFrame(rules_array, columns=['V1'])
            rules_rulemaptable = rules_rulemaptable[rules_rulemaptable != "NONE"]
            colnames_df = pd.DataFrame(rulepreds.columns,columns=['V2'])
            rules_rulemaptable = pd.concat([rules_rulemaptable, colnames_df], axis=1)
            rules_support.index = rules_rulemaptable.index
            rules_rulemaptable['rules_support'] = rules_support
            rules_scale.index = rules_rulemaptable.index
            rules_rulemaptable['rules_scale'] = rules_scale
            uniquerules = rules_rulemaptable.drop_duplicates('V1')['V2']
            rulepreds = rulepreds[uniquerules]
            rules_rulemaptable.columns = ["rule", "term", "support", "scale"]
        else:
            rules = None
            rules_rulemaptable =None
        if add_splines:
            threshold = 10
            print("training splines")
            traindata_s = pd.concat([traindata.loc[:, traindata.apply(lambda col: len(col.unique())) > threshold], traindata['dependent']], axis=1)
            traindata_s.rename(columns={traindata_s.columns[-1]: "dependent"}, inplace=True)
            # to make sure we only estimate splines for variables where it makes sense
            splines = gamsplines(traindata_s, df=splines_df, gamtype="gam", threshold=threshold, splinetype=spline_type)
            splinepreds = predict_gamsplines(splines,valdata)
            splines_stddevs = np.zeros((splinepreds.shape[1], 1))
            for j in range(splinepreds.shape[1]):
                stddev = splinepreds.iloc[:, j].std()
                splines_stddevs[j] = 1 if stddev == 0 else stddev
            splinepreds_array = splinepreds.values
            splines_stddevs = splines_stddevs.flatten()
            splines_valdata = 0.4 * (splinepreds_array / splines_stddevs)
            splines_valdata = pd.DataFrame(splines_valdata, columns=splinepreds.columns)
        else:
            splines = None
            splines_stddevs = None

        if add_lterms:
        # Standart sapmaları hesapla
            lterms_stddevs = np.array([traindata.iloc[:, j].std() if traindata.iloc[:, j].std() != 0 else 1 for j in range(p)])

            # valdata'yı normalize et ve 0.4 ile çarp
            lterms_valdata = 0.4 * (valdata.iloc[:, :p].values / lterms_stddevs)
            lterms_valdata = pd.DataFrame(lterms_valdata, columns=traindata.columns[:p])
            # Sütun isimlerini al
            lterms = traindata.columns[:p]
        else:
            lterms_stddevs = None
            lterms = None
    if isinstance(termmodel, dict):
        if add_rules:
            n = len(valdata.iloc[:, 0])
            ts = termmodel['rules.rulemaptable'].shape[0]
            member_pred = pd.DataFrame(0, index=range(n), columns=range(ts))
            ts = 0
            flat_list = [item for sublist in termmodel['rules'] for item in sublist]
            for m in range(termmodel['rules.nrtrees']):
                if termmodel['rules'][m] != "NONE":
                    print(f"extracting rules for tree {m + 1}")
                    # Burada R'daki predict.rpart_rules fonksiyonunun Python versiyonunu kullanmalısınız
                    pred_prob_m = predict_rpart_rules(termmodel['rules'][m], valdata)
                    nrr = pred_prob_m.shape[1]
                    member_pred.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
                    new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
                    current_column_names = member_pred.columns.tolist()
                    current_column_names[ts:ts+nrr] = new_column_names
                    member_pred.columns = current_column_names
                    ts += nrr

            rulepreds = member_pred
            termmodel['rules.rulemaptable']['normalized'] = termmodel['rules.rulemaptable']['rule'].apply(normalize_logic)
            tmp_rulesmaptable_unique = termmodel['rules.rulemaptable'].drop_duplicates(subset='normalized').copy()
            tmp_rulesmaptable_unique.drop(columns=['normalized'], inplace=True)
            termmodel['rules.rulemaptable']=tmp_rulesmaptable_unique.copy()
            rules_array = np.array(termmodel['rules.rulemaptable']['rule']).reshape(-1, 1)
            #rules_array = np.array(termmodel['rules.rulemaptable']['rule'][termmodel['rules.rulemaptable']['support'] >= 0.1]).reshape(-1, 1)
            rules_rulemaptable = pd.DataFrame(rules_array, columns=['V1'])
            term_array = np.array(termmodel['rules.rulemaptable']['term']).reshape(-1, 1)
            ##term_array = np.array(termmodel['rules.rulemaptable']['term'][termmodel['rules.rulemaptable']['support'] >= 0.1]).reshape(-1, 1)
            flat_terms = [t[0] for t in term_array]
            columns_to_keep = [col for col in flat_terms if col in rulepreds.columns]
            rulepreds = rulepreds[columns_to_keep]
            rules_rulemaptable = rules_rulemaptable[rules_rulemaptable != "NONE"]
            
            colnames_df = pd.DataFrame(rulepreds.columns,columns=['V2'])
            rules_rulemaptable = pd.concat([rules_rulemaptable, colnames_df], axis=1)
            uniquerules = rules_rulemaptable.drop_duplicates('V1')['V2']
            uniquerules = [re.sub(r"_.*", "", rule) for rule in uniquerules]
            rulepreds = rulepreds[uniquerules]
            rulepreds.columns = uniquerules

        if add_splines:
            splinepreds = predict_gamsplines(termmodel['splines'], valdata)
            splinepreds = 0.4 * (splinepreds / np.array(termmodel['splines.stddevs']))
            print("splines was added")

        if add_lterms:
            #ltermpreds = valdata.iloc[:, :len(termmodel['lterms.stddevs'])]
            ltermpreds = 0.4 * (valdata.iloc[:, :len(termmodel['lterms.stddevs'])] / np.array(termmodel['lterms.stddevs']))
            print("lterms was added")
        #rules_rulemaptable= termmodel['rules.rulemaptable'][termmodel['rules.rulemaptable']['support'] >= 0.1]
        rules_rulemaptable = termmodel['rules.rulemaptable']
        rules_trees = termmodel['rules.trees']
        splines = termmodel['splines']
        splines_stddevs = termmodel['splines.stddevs']
        lterms = termmodel['lterms']
        lterms_stddevs = termmodel['lterms.stddevs']
        rules = termmodel['rules']
        rules_nrtrees = termmodel['rules.nrtrees']
        #rulepreds = termmodel['terms.rules']
        splinepreds = termmodel['terms.splines']
        

    # Koşullara göre veri setlerini birleştir
    if add_lterms and add_rules and add_splines:
       termselmatrix = pd.concat([rulepreds, splinepreds, ltermpreds, valdata.iloc[:, -1]], axis=1)
    elif add_lterms and add_rules:
        termselmatrix = pd.concat([rulepreds, ltermpreds, valdata.iloc[:, -1]], axis=1)
    elif add_rules:
        termselmatrix = pd.concat([rulepreds, valdata.iloc[:, [-1]]], axis=1)
    else:
        print("Rule ensemble without rules doesn't make sense. Computation halted")
        exit()
    droppedvariables = ""
    sgl_indexes = None
    if regular_type not in ["SSVS", "caret.optimized", "SGL", "SGL2","lasso2"]:
        print("Perform Lasso/Ridge/GLMnet regularization")
        y = pd.factorize(termselmatrix.iloc[:, -1])[0]  # Son sütunun faktörize edilmiş hali
        x = termselmatrix.iloc[:, :-1].values  # Son sütunu hariç tüm sütunlar
        if cv_metric == "AUC":
            cv_metric = 'auc'
        elif cv_metric == "-LL":
            cv_metric = 'deviance'
        ##system.time(regular.model<-cv.glmnet(x,y,family="binomial",nfolds=cv.folds,alpha=regular.alpha,parallel=TRUE,type.measure=cv_metric,nlambda=nlambda))
        y_df=pd.DataFrame(y)
        y_df = y_df.astype(np.float64)
        y = y_df.to_numpy()
        y = y.reshape(-1, 1)
        x_df= pd.DataFrame(x)
        x = x_df.to_numpy()
        ##model = cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial',nfolds=cv_folds,alpha=regular_alpha,parallel=True,measure="AUC",)
        robjects.globalenv['x'] = numpy2ri.py2rpy(x)
        robjects.globalenv['y'] = numpy2ri.py2rpy(y)
        robjects.globalenv['cv_folds'] = cv_folds
        robjects.globalenv['regular_alpha'] = regular_alpha
        robjects.globalenv['cv_metric'] = cv_metric
        robjects.globalenv['nlambda'] = nlambda
        robjects.globalenv['parallel'] = parallel
        robjects.globalenv['nr_cores'] = nr_cores
        if parallel:
            robjects.r('''registerDoParallel(cores=nr_cores)''')
            robjects.r('''system.time(regular.model <- cv.glmnet(x, y, family="binomial", nfolds=cv_folds, alpha=regular_alpha, parallel=TRUE, type.measure=cv_metric, nlambda=nlambda))''')
        else:
            robjects.r('''regular.model <- cv.glmnet(x, y, family="binomial", nfolds=cv_folds, alpha=regular_alpha, type.measure=cv_metric, nlambda=nlambda)''')

        regular_model = robjects.globalenv['regular.model']
        selected = (regular_model.rx2('lambda') == regular_model.rx2('lambda.1se')).astype(int)

        lambdaseloverview = pd.DataFrame({
            'lambda': regular_model.rx2('lambda'),
            'lldiff': regular_model.rx2('cvm'),  # cv.glmnet'in döndürdüğü ikinci değere karşılık gelir
            'llSD': regular_model.rx2('cvsd'),  # cv.glmnet'in döndürdüğü üçüncü değere karşılık gelir
            'nrcoefs': regular_model.rx2('nzero'),  # Katsayı sayısı
            'selection': selected})
        lambdaseloverview = lambdaseloverview.T
        lambdaseloverview.index = ['lambda', 'lldiff', 'llSD', 'nrcoefs', 'selection']
        robjects.r('''regular.coefficients <- predict(regular.model$glmnet.fit, type="coefficients", s=regular.model$lambda.1se)
                      regular.coefficients<-as.matrix(regular.coefficients)''')
        regular_coefficients = robjects.r['regular.coefficients']
        column_names = termselmatrix.columns[:-1].tolist()
        column_names.insert(0, 'Intercept')
        column_names_df = pd.DataFrame(column_names, columns=['V2'])
        regular_coefficients=pd.DataFrame(regular_coefficients, columns=['V1'])
        regular_coefficients = pd.concat([regular_coefficients, column_names_df], axis=1)
        filtered_data = regular_coefficients[abs(regular_coefficients['V1']) > 0]
        # Filtrelenmiş veriyi 'beta' ve 'term' kolon isimleriyle yeni bir DataFrame'e dönüştür
        regular_coefficients_selected = filtered_data.rename(columns={'V1': 'beta', 'V2': 'term'}).reset_index(drop=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, rules_rulemaptable, on='term', how='left')
        a = pd.DataFrame({'term': lterms, 'var_sd': lterms_stddevs})
        b = pd.DataFrame({'term': splines['smoothterms']['term'], 'var_sd': splines_stddevs})
        c = pd.concat([a, b], ignore_index=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, c, on='term', how='left')
        regular_coefficients_selected['termtype'] = 'lterm'  # Default
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('rule'), 'termtype'] = 'rule'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('s('), 'termtype'] = 'spline'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('Inter'), 'termtype'] = 'intercept'
        regular_coefficients_selected.fillna('na', inplace=True)

        cols_to_convert = [0,2,4,5]  # Assuming these are the correct column indices
        regular_coefficients_selected.iloc[:, cols_to_convert] = regular_coefficients_selected.iloc[:, cols_to_convert].apply(pd.to_numeric, errors='coerce')
        #for term in range(len(regular_coefficients_selected)):
            #if regular_coefficients_selected.loc[term, 'termtype'] == 'rule':
                #regular_coefficients_selected.loc[term, 'imp'] = abs(regular_coefficients_selected.loc[term, 'beta']) * np.sqrt(regular_coefficients_selected.loc[term, 'support'] * (1 - regular_coefficients_selected.loc[term, 'support']))
            #elif regular_coefficients_selected.loc[term, 'termtype'] == 'lterm':
                #regular_coefficients_selected.loc[term, 'imp'] = abs(regular_coefficients_selected.loc[term, 'beta']) * regular_coefficients_selected.loc[term, 'var_sd']
            #elif regular_coefficients_selected.loc[term, 'termtype'] == 'spline':
                #regular_coefficients_selected.loc[term, 'imp'] = abs(regular_coefficients_selected.loc[term, 'beta']) * regular_coefficients_selected.loc[term, 'var_sd']
        imp = np.zeros(len(regular_coefficients_selected))
        for i, row in regular_coefficients_selected.iterrows():
            if row['termtype'] == 'rule':
                imp[i] = abs(row['beta']) * np.sqrt(row['support'] * (1 - row['support']))
            elif row['termtype'] in ['lterm', 'spline']:
                imp[i] = abs(row['beta']) * row['var_sd']

        imp_normalized = imp / max(imp) * 100
        regular_coefficients_selected['imp'] = imp_normalized
    elif regular_type == "SGL2":
        pred ={}
        y = pd.factorize(termselmatrix.iloc[:, -1])[0]  # Son sütunun faktörize edilmiş hali
        ##y = (y == depvalues[1]).astype(int)
        x = termselmatrix.iloc[:, :-1] # Son sütunu hariç tüm sütunlar
        if sgl_reduce_ruleset:
            #Remove duplicate rules based on the first column of rules_rulemaptable
            duprules = rules_rulemaptable[rules_rulemaptable.duplicated(subset=[rules_rulemaptable.columns[0]])]
            x = x.drop(columns=duprules.term, errors='ignore')
            ##x = np.delete(x, duprules, axis=1)
            keptrules = rules_rulemaptable.index[~rules_rulemaptable.duplicated(subset=[rules_rulemaptable.columns[0]])]
            # Remove equivalent (mutually exclusive) rules
            # Assuming the transformations are directly applicable, adjust if your rule logic differs
            tmp_rulesmaptable = rules_rulemaptable.loc[keptrules].copy()
            tmp_rulesmaptable[tmp_rulesmaptable.columns[0]] = tmp_rulesmaptable[tmp_rulesmaptable.columns[0]].str.replace("<", ">=", regex=True)
            duprules = tmp_rulesmaptable[tmp_rulesmaptable.duplicated(subset=[tmp_rulesmaptable.columns[0]])]
            x = x.drop(columns=duprules.term, errors='ignore')
            keptrules = tmp_rulesmaptable.index[~tmp_rulesmaptable.duplicated(subset=[tmp_rulesmaptable.columns[0]])]
            # Reduce rulemaptable to keep only the unique and non-equivalent rules
            rules_rulemaptable = rules_rulemaptable.loc[keptrules]

        if sgl_grouped_ruleset:

            robjects.globalenv['rules.rulemaptable'] = rules_rulemaptable
            testp=robjects.r('''
            testp <- gsub(" (-)?[0-9]+(\\\\.[0-9]+)?", " 9999999", rules.rulemaptable[,1], perl=TRUE)
            testp <- gsub("(>|<)=?", "", testp, perl=TRUE)
            testp <- as.data.frame(testp)
            colnames(testp) <- "testp"
            reorder_idx <- order(rules.rulemaptable[,1])
            testp <- testp[reorder_idx,,drop=FALSE]
            testp <- transform(testp, id=as.numeric(factor(testp)))''')
            firstvars = testp['testp'].str.extract(r"indata\['([^']*)'\]").fillna('')
            robjects.globalenv['firstvars'] = firstvars
            robjects.globalenv['testp'] = testp
            # Sayısal değerleri 9999999 ile değiştir
            testp = rules_rulemaptable
            testp["rule"] = testp["rule"].replace(to_replace=r"(-)?[0-9]+(\.[0-9]+)?", value="9999999", regex=True)
            # Karşılaştırma operatörlerini kaldır
            testp = testp.replace(to_replace=r"(>|<)=?", value="", regex=True)
            testp["rule"] = testp["rule"].replace(to_replace=r"(>|<)=?", value="", regex=True)

            # Yeni DataFrame oluştur ve sırala

            testp.rename(columns={'rule': 'testp'}, inplace=True)
            reorder_idx = rules_rulemaptable.iloc[:, 0].argsort()
            testp = testp.iloc[reorder_idx]
            testp['id'] = pd.factorize(testp['testp'])[0] + 1

            # İlk değişken isimlerini çıkar
            firstvars = testp['testp'].str.extract(r"indata\['([^']*)'\]").fillna('')
            firstvars[~testp['testp'].str.contains("indata")] = ""
            testp['firstvars'] = firstvars
            testp['fullname'] = testp['term'].astype(str) + "_" + testp['firstvars']
            testp['suffix'] = testp.groupby('id')['fullname'].transform(lambda x: '_' + x.iloc[0] if len(x) == 1 else '')
            # rules.rulemaptable güncelle
            rules_rulemaptable = rules_rulemaptable.reset_index(drop=True)
            testp = testp.reset_index(drop=True)
            rules_rulemaptable['term'] = testp['fullname'] + testp['suffix']
            rules_rulemaptable = rules_rulemaptable.iloc[reorder_idx.argsort()]
        if cv_metric == "AUC":
            scoring = 'roc_auc'
        elif cv_metric == "-LL":
            scoring = 'neg_log_loss'
        non_constant_columns = x.columns[x.nunique() != 1]
        x = x[non_constant_columns]
        dropped_variables = x.columns[x.nunique() == 1]
        indexes = np.ones((x.shape[1], 1))
        constant_columns = x.columns[x.nunique() == 1]
        dropped_variables = constant_columns
        grpcnt = 0
        indexes = [0] * x.shape[1]  # x veri setinin sütun sayısı kadar 0'larla dolu bir liste oluştur
        robjects.globalenv['x'] = x
        robjects.globalenv['sgl.grouped.ruleset'] = sgl_grouped_ruleset
        robjects.globalenv['traindata'] = traindata
        r_code = """
        indexes <- mat.or.vec(ncol(x), 1)
        grpcnt <- 0
        for (i in 1:ncol(x)) {
          if (sgl.grouped.ruleset == TRUE) {
            # Özgün eğitim veri setindeki değişken isimleriyle ve birleştirilmemiş kural isimleriyle eşleşen terimleri tanımlayın
            matchidx_condition <- grepl(sub("rule[0-9]+_", "", sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i]))),
                                        c(colnames(traindata), colnames(x)[which(grepl("rule[0-9]+", colnames(x), perl=TRUE) & !grepl("_", colnames(x), perl=TRUE))])) * 1
          } else {
            matchidx_condition <- grepl(sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i])), colnames(traindata)) * 1
          }
          matchidx <- (grepl(sub("rule[0-9]+_", "", sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i]))), colnames(x))) * 1
          if (indexes[i] == 0) {
            grpcnt <- grpcnt + 1
            if (sum(matchidx) > 0 && sum(matchidx_condition) > 0) {
              indexes <- indexes + matchidx * grpcnt
            } else {
              indexes[i] <- grpcnt
            }
          }
        }
        sgl.indexes<-cbind(colnames(x),indexes)
        """
        robjects.r(r_code)
        # for i, col_name in enumerate(x.columns):
        #     if sgl_grouped_ruleset:
        #         # Düzenli ifadelerle sütun isimlerinden gerekli kısımları çıkar
        #         modified_col_name = re.sub(r"rule[0-9]+_", "", re.sub(r"\)", "", re.sub(r"s\(", "", col_name)))
        #         # Eğitim verisi ve belirli kural isimleriyle eşleşme koşulu
        #         match_condition = int(any(re.search(modified_col_name, name) for name in list(traindata.columns) + [x for x in x.columns if re.search(r"rule[0-9]+", x) and not "_" in x]))
        #     else:
        #         modified_col_name = re.sub(r"\)", "", re.sub(r"s\(", "", col_name))
        #         match_condition = int(any(re.search(modified_col_name, name) for name in traindata.columns))
        #     matchidx = int(re.search(modified_col_name, col_name) is not None)

        #     if indexes[i] == 0:
        #         grpcnt += 1
        #         if matchidx > 0 and match_condition > 0:
        #             indexes = [index + matchidx * grpcnt if index == 0 else index for index in indexes]
        #         else:
        #             indexes[i] = grpcnt
        # sgl_indexes = pd.DataFrame({'colnames': x.columns, 'indexes': indexes})
        ##system.time(regular.model<-cv.glmnet(x,y,family="binomial",nfolds=cv.folds,alpha=regular.alpha,parallel=TRUE,type.measure=cv_metric,nlambda=nlambda))
        y_df=pd.DataFrame(y)
        y_df = y_df.astype(np.float64)
        y = y_df.to_numpy()
        y = y.reshape(-1, 1)
        x_df= pd.DataFrame(x)
        x = x_df.to_numpy()
        ##model = cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial',nfolds=cv_folds,alpha=regular_alpha,parallel=True,measure="AUC",)
        robjects.globalenv['x'] = numpy2ri.py2rpy(x)
        robjects.globalenv['y'] = numpy2ri.py2rpy(y)
        robjects.globalenv['cv_folds'] = cv_folds
        robjects.globalenv['regular_alpha'] = regular_alpha
        robjects.globalenv['cv_metric'] = cv_metric
        robjects.globalenv['nlambda'] = nlambda
        robjects.globalenv['parallel'] = parallel
        robjects.globalenv['nr_cores'] = nr_cores
        robjects.globalenv['sgl_alpha'] = sgl_alpha
        robjects.globalenv['maxit'] = maxit
        robjects.globalenv['standardize'] = standardize

        if parallel:
            robjects.r('''registerDoParallel(cores=nr_cores)''')
            robjects.r('''regular.model<-cv.oem(x,y,family="binomial",nfolds=cv_folds,alpha=regular_alpha,parallel=TRUE,ncores=nr_cores,type.measure=cv_metric,nlambda=nlambda,penalty="sparse.grp.lasso",groups=indexes,tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
        else:
            robjects.r('''regular.model<-cv.oem(x,y,family="binomial",nfolds=cv_folds,alpha=regular_alpha,type.measure=cv_metric,nlambda=nlambda,penalty="sparse.grp.lasso",groups=indexes,tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
        regular_model = robjects.globalenv['regular.model']
        for i in range(1, len(regular_model) + 1):
            if isinstance(regular_model[i][0], list) and len(regular_model[i][0]) == 1:
                regular_model[i][0] = regular_model[i][0]
        selected = (regular_model.rx2('lambda')[0] == regular_model.rx2('lambda.1se')[0]).astype(int)
        lambdaseloverview = pd.DataFrame({
        'lambda': regular_model.rx2('lambda')[0],
        'lldiff': regular_model.rx2('cvm')[0],  # cv.glmnet'in döndürdüğü ikinci değere karşılık gelir
        'llSD': regular_model.rx2('cvsd')[0],  # cv.glmnet'in döndürdüğü üçüncü değere karşılık gelir
        'nrcoefs': regular_model.rx2('nzero')[0],  # Katsayı sayısı
        'selection': selected})
        lambdaseloverview = lambdaseloverview.T
        lambdaseloverview.index = ['lambda', 'lldiff', 'llSD', 'nrcoefs', 'selection']
        robjects.r('''regular.coefficients <- predict(regular.model$oem.fit, type="coefficients", s=regular.model$lambda.1se)
                  regular.coefficients<-as.matrix(regular.coefficients)''')
        regular_coefficients = robjects.r['regular.coefficients']
        column_names = x_df.columns.tolist()
        column_names.insert(0, 'Intercept')
        column_names_df = pd.DataFrame(column_names, columns=['V2'])
        regular_coefficients=pd.DataFrame(regular_coefficients, columns=['V1'])
        regular_coefficients = pd.concat([regular_coefficients, column_names_df], axis=1)
        filtered_data = regular_coefficients[abs(regular_coefficients['V1']) > 0]
        # Filtrelenmiş veriyi 'beta' ve 'term' kolon isimleriyle yeni bir DataFrame'e dönüştür
        regular_coefficients_selected = filtered_data.rename(columns={'V1': 'beta', 'V2': 'term'}).reset_index(drop=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, rules_rulemaptable, on='term', how='left')
        a = pd.DataFrame({'term': lterms, 'var_sd': lterms_stddevs})
        b = pd.DataFrame({'term': splines['smoothterms']['term'], 'var_sd': splines_stddevs})
        c = pd.concat([a, b], ignore_index=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, c, on='term', how='left')
        regular_coefficients_selected['termtype'] = 'lterm'  # Default
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('rule'), 'termtype'] = 'rule'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('s('), 'termtype'] = 'spline'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('Inter'), 'termtype'] = 'intercept'
        regular_coefficients_selected.fillna('na', inplace=True)

        cols_to_convert = [0,2,4,5]  # Assuming these are the correct column indices
        regular_coefficients_selected.iloc[:, cols_to_convert] = regular_coefficients_selected.iloc[:, cols_to_convert].apply(pd.to_numeric, errors='coerce')
        imp = np.zeros(len(regular_coefficients_selected))
        for i, row in regular_coefficients_selected.iterrows():
            if row['termtype'] == 'rule':
                imp[i] = abs(row['beta']) * np.sqrt(row['support'] * (1 - row['support']))
            elif row['termtype'] in ['lterm', 'spline']:
                imp[i] = abs(row['beta']) * row['var_sd']

        imp_normalized = imp / max(imp) * 100
        regular_coefficients_selected['imp'] = imp_normalized
    elif regular_type == "lasso2":
        print("lasso2 added")
        pred ={}
        y_df23 = termselmatrix['dependent'].map({'cl0':0, 'cl1':1})
        y=np.array(y_df23)  # Son sütunun faktörize edilmiş hali
        ##y = (y == depvalues[1]).astype(int)
        x = termselmatrix.iloc[:, :-1] # Son sütunu hariç tüm sütunlar
        if sgl_reduce_ruleset:
            #Remove duplicate rules based on the first column of rules_rulemaptable
            duprules = rules_rulemaptable[rules_rulemaptable.duplicated(subset=[rules_rulemaptable.columns[0]])]
            x = x.drop(columns=duprules.term, errors='ignore')
            ##x = np.delete(x, duprules, axis=1)
            keptrules = rules_rulemaptable.index[~rules_rulemaptable.duplicated(subset=[rules_rulemaptable.columns[0]])]
            # Remove equivalent (mutually exclusive) rules
            # Assuming the transformations are directly applicable, adjust if your rule logic differs
            tmp_rulesmaptable = rules_rulemaptable.loc[keptrules].copy()
            tmp_rulesmaptable[tmp_rulesmaptable.columns[0]] = tmp_rulesmaptable[tmp_rulesmaptable.columns[0]].str.replace("<", ">=", regex=True)
            duprules = tmp_rulesmaptable[tmp_rulesmaptable.duplicated(subset=[tmp_rulesmaptable.columns[0]])]
            x = x.drop(columns=duprules.term, errors='ignore')
            keptrules = tmp_rulesmaptable.index[~tmp_rulesmaptable.duplicated(subset=[tmp_rulesmaptable.columns[0]])]
            # Reduce rulemaptable to keep only the unique and non-equivalent rules
            rules_rulemaptable = rules_rulemaptable.loc[keptrules]

        if sgl_grouped_ruleset:

            robjects.globalenv['rules.rulemaptable'] = rules_rulemaptable
            testp=robjects.r('''
            testp <- gsub(" (-)?[0-9]+(\\\\.[0-9]+)?", " 9999999", rules.rulemaptable[,1], perl=TRUE)
            testp <- gsub("(>|<)=?", "", testp, perl=TRUE)
            testp <- as.data.frame(testp)
            colnames(testp) <- "testp"
            reorder_idx <- order(rules.rulemaptable[,1])
            testp <- testp[reorder_idx,,drop=FALSE]
            testp <- transform(testp, id=as.numeric(factor(testp)))''')
            firstvars = testp['testp'].str.extract(r"indata\['([^']*)'\]").fillna('')
            robjects.globalenv['firstvars'] = firstvars
            robjects.globalenv['testp'] = testp
            # Sayısal değerleri 9999999 ile değiştir
            testp = rules_rulemaptable
            testp["rule"] = testp["rule"].replace(to_replace=r"(-)?[0-9]+(\.[0-9]+)?", value="9999999", regex=True)
            # Karşılaştırma operatörlerini kaldır
            testp = testp.replace(to_replace=r"(>|<)=?", value="", regex=True)
            testp["rule"] = testp["rule"].replace(to_replace=r"(>|<)=?", value="", regex=True)

            # Yeni DataFrame oluştur ve sırala

            testp.rename(columns={'rule': 'testp'}, inplace=True)
            reorder_idx = rules_rulemaptable.iloc[:, 0].argsort()
            testp = testp.iloc[reorder_idx]
            testp['id'] = pd.factorize(testp['testp'])[0] + 1

            # İlk değişken isimlerini çıkar
            firstvars = testp['testp'].str.extract(r"indata\['([^']*)'\]").fillna('')
            firstvars[~testp['testp'].str.contains("indata")] = ""
            testp['firstvars'] = firstvars
            testp['fullname'] = testp['term'].astype(str) + "_" + testp['firstvars']
            testp['suffix'] = testp.groupby('id')['fullname'].transform(lambda x: '_' + x.iloc[0] if len(x) == 1 else '')
            # rules.rulemaptable güncelle
            rules_rulemaptable = rules_rulemaptable.reset_index(drop=True)
            testp = testp.reset_index(drop=True)
            rules_rulemaptable['term'] = testp['fullname'] + testp['suffix']
            rules_rulemaptable = rules_rulemaptable.iloc[reorder_idx.argsort()]
        if cv_metric == "AUC":
            cv_metric = 'auc'
        elif cv_metric == "-LL":
            cv_metric = 'deviance'
        non_constant_columns = x.columns[x.nunique() != 1]
        x = x[non_constant_columns]
        dropped_variables = x.columns[x.nunique() == 1]
        indexes = np.ones((x.shape[1], 1))
        constant_columns = x.columns[x.nunique() == 1]
        dropped_variables = constant_columns
        grpcnt = 0
        indexes = [0] * x.shape[1]  # x veri setinin sütun sayısı kadar 0'larla dolu bir liste oluştur
        robjects.globalenv['x'] = x
        robjects.globalenv['sgl.grouped.ruleset'] = sgl_grouped_ruleset
        robjects.globalenv['traindata'] = traindata
        r_code = """
        indexes <- mat.or.vec(ncol(x), 1)
        grpcnt <- 0
        for (i in 1:ncol(x)) {
          if (sgl.grouped.ruleset == TRUE) {
            # Özgün eğitim veri setindeki değişken isimleriyle ve birleştirilmemiş kural isimleriyle eşleşen terimleri tanımlayın
            matchidx_condition <- grepl(sub("rule[0-9]+_", "", sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i]))),
                                        c(colnames(traindata), colnames(x)[which(grepl("rule[0-9]+", colnames(x), perl=TRUE) & !grepl("_", colnames(x), perl=TRUE))])) * 1
          } else {
            matchidx_condition <- grepl(sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i])), colnames(traindata)) * 1
          }
          matchidx <- (grepl(sub("rule[0-9]+_", "", sub("\\\\)", "", sub("s\\\\(", "", colnames(x)[i]))), colnames(x))) * 1
          if (indexes[i] == 0) {
            grpcnt <- grpcnt + 1
            if (sum(matchidx) > 0 && sum(matchidx_condition) > 0) {
              indexes <- indexes + matchidx * grpcnt
            } else {
              indexes[i] <- grpcnt
            }
          }
        }
        sgl.indexes<-cbind(colnames(x),indexes)
        """
        robjects.r(r_code)
        # for i, col_name in enumerate(x.columns):
        #     if sgl_grouped_ruleset:
        #         # Düzenli ifadelerle sütun isimlerinden gerekli kısımları çıkar
        #         modified_col_name = re.sub(r"rule[0-9]+_", "", re.sub(r"\)", "", re.sub(r"s\(", "", col_name)))
        #         # Eğitim verisi ve belirli kural isimleriyle eşleşme koşulu
        #         match_condition = int(any(re.search(modified_col_name, name) for name in list(traindata.columns) + [x for x in x.columns if re.search(r"rule[0-9]+", x) and not "_" in x]))
        #     else:
        #         modified_col_name = re.sub(r"\)", "", re.sub(r"s\(", "", col_name))
        #         match_condition = int(any(re.search(modified_col_name, name) for name in traindata.columns))
        #     matchidx = int(re.search(modified_col_name, col_name) is not None)

        #     if indexes[i] == 0:
        #         grpcnt += 1
        #         if matchidx > 0 and match_condition > 0:
        #             indexes = [index + matchidx * grpcnt if index == 0 else index for index in indexes]
        #         else:
        #             indexes[i] = grpcnt
        # sgl_indexes = pd.DataFrame({'colnames': x.columns, 'indexes': indexes})
        ##system.time(regular.model<-cv.glmnet(x,y,family="binomial",nfolds=cv.folds,alpha=regular.alpha,parallel=TRUE,type.measure=cv_metric,nlambda=nlambda))
        y_df=pd.DataFrame(y)
        y_df = y_df.astype(np.float64)
        y = y_df.to_numpy()
        y = y.reshape(-1, 1)
        x_df= pd.DataFrame(x)
        x = x_df.to_numpy()
        combined = x_df.join(y_df)
        # if(add_stability):
        #     n_rules = rulepreds.shape[1]
        #     stabilities = [
        #         np.mean([
        #             sorensen_dice_index2(rulepreds.iloc[:, j], rulepreds.iloc[:, i])
        #             for i in range(n_rules) if i != j
        #             ])
        #         for j in range(n_rules)
        #         ]
        #     df_stabilities = pd.DataFrame(stabilities, columns=['V1'])
        #     colnames_df_stabilities =pd.DataFrame(rulepreds.columns,columns=['term'])
        #     df_stabilities = pd.concat([df_stabilities, colnames_df_stabilities], axis=1)
        #     df_stabilities.columns = ["sindex", "term"]
        #     df_stabilities.index = rules_rulemaptable.index
        #     e_index=df_stabilities.shape[0]
        #     stability_index = np.array(df_stabilities['sindex'].head(e_index))
        #     #stability_index = 1 - stability_index
        #     robjects.globalenv['stability_index'] = stability_index
        #     robjects.globalenv['s_index'] = 1
        #     robjects.globalenv['e_index'] = e_index
        #     robjects.globalenv['eta_stability'] = 0.01
        
        if(add_stability):
            # n = len(traindata)
            # total_rules = sum(len(rule_set) for rule_set in model['rules'])
            # member_pred_for_stability = pd.DataFrame(np.zeros((n, total_rules)))
            # ts = 0
            # cnt = 0
            # for m in range(10):
            #     if "NONE" not in model['rules'][m]:
            #         pred_prob_m = predict_rpart_rules(model['rules'][m], traindata,predvarprefix="rule")
            #         nrr = pred_prob_m.shape[1]
            #         cnt += 1
            #         member_pred_for_stability.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
            #         new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
            #         current_column_names = member_pred_for_stability.columns.tolist()
            #         current_column_names[ts:ts+nrr] = new_column_names
            #         member_pred_for_stability.columns = current_column_names
            #         ts += nrr
            # print(f"extracting bitti")
            rulepreds_for_stability = member_pred
            n_rules = rulepreds_for_stability.shape[1]
            stabilities = [
                np.mean([
                    sorensen_dice_index2(rulepreds_for_stability.iloc[:, j], rulepreds_for_stability.iloc[:, i])
                    for i in range(n_rules) if i != j
                    ])
                for j in range(n_rules)
                ]
            df_stabilities = pd.DataFrame(stabilities, columns=['V1'])
            colnames_df_stabilities =pd.DataFrame(rulepreds_for_stability.columns,columns=['term'])
            df_stabilities = pd.concat([df_stabilities, colnames_df_stabilities], axis=1)
            df_stabilities.columns = ["sindex", "term"]
            df_stabilities = df_stabilities.loc[rules_rulemaptable.index]
            e_index=df_stabilities.shape[0]
            stability_index = np.array(df_stabilities['sindex'].head(e_index))
            stability_index = 1 - stability_index
            robjects.globalenv['stability_index'] = stability_index
            robjects.globalenv['s_index'] = 1
            robjects.globalenv['e_index'] = e_index
            robjects.globalenv['eta_stability'] = 0.001
 
   
        # if(add_stability):
        #     stabilities = termmodel['stability']
        #     stabilities = stabilities.loc[rules_rulemaptable.index]
        #     e_index=stabilities.shape[0]
        #     stability_index = np.array(stabilities['sindex'].head(e_index))
        #     stability_index = 1 - stability_index
        #     robjects.globalenv['stability_index'] = stability_index
        #     robjects.globalenv['s_index'] = 1
        #     robjects.globalenv['e_index'] = e_index
        #     robjects.globalenv['eta_stability'] = 0.02
        ##model = cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial',nfolds=cv_folds,alpha=regular_alpha,parallel=True,measure="AUC",)
        robjects.globalenv['x'] = numpy2ri.py2rpy(x)
        robjects.globalenv['y'] = numpy2ri.py2rpy(y)
        robjects.globalenv['cv_folds'] = cv_folds
        robjects.globalenv['regular_alpha'] = regular_alpha
        robjects.globalenv['cv_metric'] = cv_metric
        robjects.globalenv['nlambda'] = nlambda
        robjects.globalenv['parallel'] = parallel
        robjects.globalenv['nr_cores'] = nr_cores
        robjects.globalenv['sgl_alpha'] = sgl_alpha
        robjects.globalenv['maxit'] = maxit
        robjects.globalenv['standardize'] = standardize
        if parallel:
            robjects.r('''registerDoParallel(cores=nr_cores)''')
            if add_stability:
                robjects.r('''regular.model<-oemEren::cv.oem(x,y,family="binomial",nfolds=cv_folds,c_stability = stability_index,eta_stability = eta_stability,s_index = 1,e_index = e_index,alpha=regular_alpha,parallel=TRUE,ncores=nr_cores,type.measure=cv_metric,nlambda=nlambda,penalty="lasso",tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
            else:
                robjects.r('''regular.model<-cv.oem(x,y,family="binomial",nfolds=cv_folds,alpha=regular_alpha,parallel=TRUE,ncores=nr_cores,type.measure=cv_metric,nlambda=nlambda,penalty="lasso",tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
        else:
            if add_stability:
                robjects.r('''regular.model<-oemEren::cv.oem(x,y,family="binomial",nfolds=cv_folds,c_stability = stability_index,eta_stability = eta_stability,s_index = 1,e_index = e_index,alpha=regular_alpha,type.measure=cv_metric,nlambda=nlambda,penalty="lasso",tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
            else:
                robjects.r('''regular.model<-cv.oem(x,y,family="binomial",nfolds=cv_folds,alpha=regular_alpha,type.measure=cv_metric,nlambda=nlambda,penalty="lasso",tau=sgl_alpha,maxit=maxit,standardize=standardize)''')
        regular_model = robjects.globalenv['regular.model']
        #for i in range(1, len(regular_model) + 1):
         #   if isinstance(regular_model[i][0], list) and len(regular_model[i][0]) == 1:
          #      regular_model[i][0] = regular_model[i][0]
        selected = (regular_model.rx2('lambda')[0] == regular_model.rx2('lambda.1se')[0]).astype(int)
        lambdaseloverview = pd.DataFrame({
        'lambda': regular_model.rx2('lambda')[0],
        'lldiff': regular_model.rx2('cvm')[0],  # cv.glmnet'in döndürdüğü ikinci değere karşılık gelir
        'llSD': regular_model.rx2('cvsd')[0],  # cv.glmnet'in döndürdüğü üçüncü değere karşılık gelir
        'nrcoefs': regular_model.rx2('nzero')[0],  # Katsayı sayısı
        'selection': selected})
        lambdaseloverview = lambdaseloverview.T
        lambdaseloverview.index = ['lambda', 'lldiff', 'llSD', 'nrcoefs', 'selection']
        robjects.r('''regular.coefficients <- predict(regular.model$oem.fit, type="coefficients", s=regular.model$lambda.1se)
                  regular.coefficients<-as.matrix(regular.coefficients)''')
        regular_coefficients = robjects.r['regular.coefficients']
        column_names = x_df.columns.tolist()
        column_names.insert(0, 'Intercept')
        column_names_df = pd.DataFrame(column_names, columns=['V2'])
        regular_coefficients=pd.DataFrame(regular_coefficients, columns=['V1'])
        regular_coefficients = pd.concat([regular_coefficients, column_names_df], axis=1)
        filtered_data = regular_coefficients[abs(regular_coefficients['V1']) > 0]
        # Filtrelenmiş veriyi 'beta' ve 'term' kolon isimleriyle yeni bir DataFrame'e dönüştür
        regular_coefficients_selected = filtered_data.rename(columns={'V1': 'beta', 'V2': 'term'}).reset_index(drop=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, rules_rulemaptable, on='term', how='left')
        a = pd.DataFrame({'term': lterms, 'var_sd': lterms_stddevs})
        b = pd.DataFrame({'term': splines['smoothterms']['term'], 'var_sd': splines_stddevs})
        c = pd.concat([a, b], ignore_index=True)
        regular_coefficients_selected = pd.merge(regular_coefficients_selected, c, on='term', how='left')
        regular_coefficients_selected['termtype'] = 'lterm'  # Default
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('rule'), 'termtype'] = 'rule'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('s('), 'termtype'] = 'spline'
        regular_coefficients_selected.loc[regular_coefficients_selected['term'].str.startswith('Inter'), 'termtype'] = 'intercept'
        regular_coefficients_selected.fillna('na', inplace=True)

        cols_to_convert = [0,2,4,5]  # Assuming these are the correct column indices
        regular_coefficients_selected.iloc[:, cols_to_convert] = regular_coefficients_selected.iloc[:, cols_to_convert].apply(pd.to_numeric, errors='coerce')
        imp = np.zeros(len(regular_coefficients_selected))
        for i, row in regular_coefficients_selected.iterrows():
            if row['termtype'] == 'rule':
                imp[i] = abs(row['beta']) * np.sqrt(row['support'] * (1 - row['support']))
            elif row['termtype'] in ['lterm', 'spline']:
                imp[i] = abs(row['beta']) * row['var_sd']

        imp_normalized = imp / max(imp) * 100
        regular_coefficients_selected['imp'] = imp_normalized
    if regular_type not in ["SGL2", "SGL","lasso2"]:
        robjects.r('''pred<-predict.glmnet(regular.model$glmnet.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
        pred = robjects.globalenv['pred']
    elif regular_type in ["SGL2", "lasso2"]:
        robjects.r('''pred<-predict(regular.model$oem.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
        pred = robjects.globalenv['pred']
    elif regular_type  in ["SGL"]:
        robjects.r(''' todrop <- colnames(x) %in% droppedvariables
                       x <- x[,todrop]
                       pred<-predictSGL(regular.model,x,regular.model$idx)''')
        pred = robjects.globalenv['pred']


    term_overlap =analyze_group_term_overlap(regular_coefficients_selected)
    ans =  {  'type': "classification",
              'add_rules':add_rules,
              'add_splines':add_splines,
              'add_lterms':add_lterms,
              'rules_rulemaptable':rules_rulemaptable,
              'rules_trees':member_trees,
              'splines':splines,
              'splines_stddevs':splines_stddevs,
              'lterms':lterms,
              'lterms_stddevs':lterms_stddevs,
              'rules':rules,
              'regular_model':regular_model,
              'regular_type':regular_type,
              'regular_coefficients_selected':regular_coefficients_selected,
              'rules_nrtrees':rules_nrtrees,
              'classes':depvalues,
              'droppedvariables':droppedvariables,
              'lambdaseloverview':lambdaseloverview,
              'term_overlap':term_overlap,
              'pred':pred,
              'rulepreds':rulepreds,
              'ltermpreds':ltermpreds,
              'splinepreds':splinepreds,
              'sgl_indexes':sgl_indexes,
              'reducedtermselmatrix':combined
               
              }

    ans = dict(ans)
    ans['type'] = "SREterms"
    return ans
def gamsplines(traindata, df="AUTO", gamtype="gam", threshold=5, splinetype=["tp", "cr", "cs", "ps", "ts"]):
    # Binary classification only!!!
    formula = "dependent ~ " + " + ".join(traindata.columns[:-1])
    cutoff = 0.5
    selection = False
    vardep = traindata['dependent']
    depvalues = np.unique(vardep)
    depvalues.sort()
    n = traindata.shape[0]
    p = traindata.shape[1] - 1
    maxdepth = len(np.unique(vardep))
    nclasses = len(depvalues)
    gammodels = []
    robjects.r('''gammodels <- list()''')
    robjects.globalenv['gammodels'] = (gammodels)



    for j in range(1, ceil((p-1)/20) + 1):
    # Veri çerçevesinin belirli sütunlarını seçme
        start_col = 20 * (j - 1)
        end_col = min(p, 20 * j)
        subset = traindata.iloc[:, start_col:end_col]

        # traindata_s'i oluşturma
        traindata_s = pd.concat([subset, pd.DataFrame((traindata.iloc[:, -1] == depvalues[1]) * 1)], axis=1)

        # Seçilen ve seçilmeyen sütunları belirleme
        sub = traindata_s.columns.isin([formula.split('~')[0].strip()])
        selectedvars = traindata_s.columns[~sub]

        # İki farklı listeyi başlatma
        rfs_linvars = []
        rfs_nparvars = []
        for varname in selectedvars:
            # Benzersiz değerlerin sayısını hesaplama
            uniq = traindata_s[varname].nunique()

            # Koşullara göre değişkenleri ilgili listelere ekleme
            if 1 < uniq <= threshold:
                rfs_linvars.append(varname)
            elif uniq > threshold:
                rfs_nparvars.append(varname)
        npar_form1 = []
        npar_form2 = ""
        if len(rfs_nparvars) > 0:
            if df != "AUTO":
                # Her bir değişken için spline tanımını oluştur
                for index, var in enumerate(rfs_nparvars):
                    mod_index = index % 5
                    spline_string = f"s({var}, bs='{splinetype[mod_index]}', k={df+1}, fx=True)"
                    npar_form1.append(spline_string)
                    # npar_form1'deki tüm stringleri birleştirerek npar_form2'yi oluştur
                    npar_form2 = " + ".join(npar_form1)
            if df == "AUTO":
                # Her bir değişken için spline tanımını oluştur
                for index, var in enumerate(rfs_nparvars):
                    mod_index = index % 5
                    spline_string = f"s({var}, bs='{splinetype[mod_index]}')"
                    npar_form1.append(spline_string)
                    # npar_form1'deki tüm stringleri birleştirerek npar_form2'yi oluştur
                    npar_form2 = " + ".join(npar_form1)
        else:
            npar_form2 = ""
        dependent_var = 'dependent'
        if len(rfs_linvars) > 0:
            lin_form = " + ".join(rfs_linvars)
            if len(rfs_nparvars) > 0:
                finalstring = f"{dependent_var} ~ {npar_form2} + {lin_form}"
            else:
                finalstring = f"{dependent_var} ~ {lin_form}"
        else:
            finalstring = f"{dependent_var} ~ {npar_form2}"
        # Model oluşturma

        robjects.globalenv['traindata_s'] = pandas2ri.py2rpy(traindata_s)
        robjects.globalenv['selection'] = selection
        robjects.globalenv['finalstring'] = finalstring
        robjects.globalenv['j'] = j
        robjects.r('''fmla <- as.formula(finalstring)''')
        robjects.r('''j''')


        if df != "AUTO":
            if gamtype == "bam":
                robjects.r('''
                           gammodels[[{j}]] <- bam(fmla, data=traindata_s, family=binomial(link="logit"),samfrac=0.1)
                           selected_model <- gammodels[[{j}]]
                           ''')
                selected_model = robjects.r['selected_model']
                gammodels.append(selected_model)
                # LogisticGAM sınıfını kullanarak GAM modeli oluşturma
                #gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9)).fit(traindata_s, traindata_s[dependent_var])
                #gammodels.append(gam)
                # 'bam' tipi için Python'da doğrudan bir karşılık yok
            else:
                robjects.r('''
                           gammodels[[{j}]] <- gam(fmla, data=traindata_s,select=selection, family=binomial(link="logit"))
                           selected_model <- gammodels[[{j}]]
                           ''')
                selected_model = robjects.r['selected_model']
                gammodels.append(selected_model)
        else:
            # 'AUTO' için Python'da özel bir işlem yok; model parametrelerini ayarlayın
            if gamtype == "bam":
                robjects.r('''
                           gammodels[[{j}]] <- bam(fmla, data=traindata_s, family=binomial(link="logit"),samfrac=0.1,method="GCV.Cp")
                           selected_model <- gammodels[[{j}]]
                           ''')
                #gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9)).fit(traindata_s, traindata_s[dependent_var])
                selected_model = robjects.r['selected_model']
                gammodels.append(selected_model)
                # 'bam' tipi için Python'da doğrudan bir karşılık yok
            else:
                robjects.r('''
                           gammodels[[{j}]] <- gam(fmla, data=traindata_s,select=selection, family=binomial(link="logit"),method="GCV.Cp")
                           selected_model <- gammodels[[{j}]]
                           ''')
                selected_model = robjects.r['selected_model']
                gammodels.append(selected_model)

        if 'smoothterms' not in locals():
            smoothterms, modelids, splineids = [], [], []
            # j değerine göre listeleri güncelleme
        if j == 1:
            smoothterms = rfs_nparvars.copy()
            modelids = [j] * len(rfs_nparvars)
            splineids = list(range(1, len(rfs_nparvars) + 1))
        else:
            smoothterms.extend(rfs_nparvars)
            modelids.extend([j] * len(rfs_nparvars))
            splineids.extend(list(range(1, len(rfs_nparvars) + 1)))
    smoothterms_df = pd.DataFrame({
    "var": smoothterms,
    "modelid": modelids,
    "splineid": splineids,
    "term": [f"s({term})" for term in smoothterms]})


    ans = {
    "gammodels": gammodels,
    "nmodels": j,
    "smoothterms": smoothterms_df }
    return ans

def extract_rpart_rules(model, type=["probs","binary"]):
    model=model["tree"]
    R.assign('model', model)
    ruleset=robjects.r('''if (is.null(model$splits)) ruleset<-"NONE"''')
    if ruleset=="NONE":
        ruleset = "NONE"
    else:
        ##splits = model.rx2('splits')
        splits = robjects.r('model$splits')
        frame_var = robjects.r('model$frame$var')
        relsplits = robjects.r('''
            relsplits <- model$splits[rownames(model$splits) %in% model$frame$var, , drop=FALSE]
            var <- rownames(relsplits)
            rownames(relsplits) <- c()
            relsplits <- as.data.frame(relsplits)
            relsplits <- cbind(var, relsplits)
        ''')


        cols = ['var', 'n', 'wt','dev','yval','complexity','ncompete','nsurrogate','change8']
        pandas_df = pd.DataFrame(model.rx2('frame'),cols)
        pandas_df = pandas_df.T
        node_column = pd.DataFrame(robjects.r['rownames'](model.rx2('frame')), columns=['node'])
        nested_data = pandas_df.iloc[:, 8].apply(pd.Series)
        expanded_data = pd.DataFrame(nested_data.iloc[:, :5].to_numpy(), columns=['yval2L1', 'yval2L2', 'yval2L3', 'yval2L4', 'yval2L5'])
        expanded_data['nodeprob'] = nested_data.iloc[:, 5]
        result_df = pd.concat([pandas_df.drop(pandas_df.columns[8], axis=1), expanded_data], axis=1)
        result_df.insert(0, 'node', node_column)
        frame=result_df
        frame['node'] = frame['node'].astype(int)
        frame = frame.sort_values('node').reset_index(drop=True)
        splitvars = frame['var'][~frame['var'].isin(['<leaf>'])]
        splitvars_n = frame.loc[~frame.duplicated(subset='var', keep='last') & (frame['var'] != '<leaf>'), 'n']
        addcol1 = pd.DataFrame({'V1': ['root'] + list(np.repeat(splitvars, 2))})
        addcol1 = addcol1[addcol1['V1'] != 'nan'].reset_index(drop=True)
        addcol2 = pd.DataFrame({'V1': [frame['n'].max()] + list(np.repeat(splitvars_n.values, 2))})
        addcol2 = addcol2[addcol2['V1'] != 'nan'].reset_index(drop=True)

        addcol1.columns = ['splitvar']
        addcol2.columns = ['n2']
        frame = pd.concat([frame, addcol1, addcol2], axis=1)
        frame = frame.rename(columns={'var':'oldvar', 'splitvar':'var'})
        rules = pd.merge(frame, relsplits, on='var')
        rules = rules[rules['n2'] == rules['count']]
        indexer = [False, True] * (len(rules) // 2) + [False] * (len(rules) % 2)
        rules.loc[indexer, 'ncat'] = -rules.loc[indexer, 'ncat']
        rules['operator'] = np.where(rules['ncat'] == 1, '>=', '<')


        selected_columns = rules[['node', 'var', 'operator', 'index', 'n']]
        yval2_columns =rules.loc[:, ['yval2L4', 'yval2L5', 'nodeprob']]
        combined_rules = pd.concat([selected_columns, yval2_columns], axis=1)
        combined_rules.columns = ["node", "var", "operator", "threshold", "n", "p_cl1", "p_cl0","none"]
        sorted_rules = combined_rules.sort_values(by='node', ascending=False)
        parents = np.zeros(len(sorted_rules), dtype=int)
        # Iterate over the rows in steps of 2 and compute parents
        for i in range(0, len(sorted_rules), 2):
            if i + 1 < len(sorted_rules):
                sum_n = sorted_rules.iloc[i]['n'] + sorted_rules.iloc[i + 1]['n']
                parent_node = sorted_rules[sorted_rules['n'] == sum_n]['node']

                if not parent_node.empty:
                    parent_value = parent_node.iloc[0]
                    parents[i] = parent_value
                    parents[i + 1] = parent_value
        sorted_rules['parents'] = parents

        ruleset = []
        for i in range(len(sorted_rules)):
            # Initialize the condition string
            conditions = []

            # Starting with the current rule
            conditions.append(f"(indata['{sorted_rules.iloc[i]['var']}'] {sorted_rules.iloc[i]['operator']} {sorted_rules.iloc[i]['threshold']})")

            parent_idx = sorted_rules.iloc[i]['parents']

            # Loop through the rest of the rules
            for j in range(i, len(sorted_rules)):
                if sorted_rules.iloc[j]['node'] == parent_idx:
                    conditions.append(f"(indata['{sorted_rules.iloc[j]['var']}'] {sorted_rules.iloc[j]['operator']} {sorted_rules.iloc[j]['threshold']})")
                    parent_idx = sorted_rules.iloc[j]['parents']

            # Combine conditions with 'and' representing the logical AND
            combined_conditions = " and ".join(conditions)

            # Finish the rule string
            p_cl1 = sorted_rules.iloc[i]['p_cl1'] if 'type' == 'probs' else 1
            p_cl0 = 1 - p_cl1

            # Convert the rule into Python's conditional expression format
            newrule = f"{p_cl1} if {combined_conditions} else {p_cl0}"

            # Add the rule to the ruleset
            ruleset.append(newrule)

    return ruleset



def predict_rpart_rules(ruleset, indata, predvarprefix=None):
    predictions = np.zeros((len(indata), len(ruleset)))
    for i in range(len(ruleset)):
        predictions[:, i] = indata.apply(lambda row: eval(ruleset[i], {'indata': row}), axis=1)
    predictions = pd.DataFrame(predictions)
    if predvarprefix is not None:
        colnames = [f"{predvarprefix}_r{i+1}" for i in range(len(ruleset))]
        predictions.columns = colnames
    return predictions

def predict_gamsplines(model, data):

    for j in range(model['nmodels']):
        # GAM modelini al
        gam_model = model['gammodels'][j]
        r_predict_gam = robjects.r['predict.gam']
        # Model üzerinden tahminleri hesapla
        pred = r_predict_gam(gam_model,data,type="terms")

        # Tahminleri bir pandas DataFrame'e dönüştür
        pred_df = robjects.conversion.rpy2py(pred)
        pred_df = pd.DataFrame(pred_df)
        pred_df.columns=model['smoothterms']['term']
        if j == 0:
            predictions = pred_df
        else:
            predictions = pd.concat([predictions, pred_df], axis=1)

    return predictions

def predict_tree(model, testdata, output_pred=True):


    tree_algorithm = model['tree_algorithm']
    vardep = testdata['dependent']
    n = len(testdata)
    tmp = np.unique(vardep)
    depvalues = tmp[np.argsort(tmp)]
    X = testdata.iloc[:, :-1]
    if tree_algorithm == "CART":
        pred_prob = model['tree'].predict_proba(X)
        pred_class = model['tree'].predict(X)

    pred_class2 = pred_class.astype(str)
    for i in range(len(np.unique(vardep))):
        pred_class2[(pred_class==i+1)] = str(depvalues[i])
    pred_class = pred_class2.tolist()

    if output_pred==True:
        pred_all = pd.DataFrame({'dependent': vardep, 'pred': pred_class,
                                 'prob0': pred_prob[:, 0], 'prob1': pred_prob[:, 1]})
    else:
        pred_all = None

    ind = (vardep != pred_class).astype(int)
    error_rate = np.sum(ind)/n
    output = {'pred': pred_all, 'error_rate': error_rate}

    return output


def predict_tree2(model, testdata, output_pred=True):
    tree_algorithm = model['tree_algorithm']
    #vardep = testdata['dependent'].values
    #n = len(testdata)
    #tmp = np.unique(vardep)
    #depvalues = np.sort(tmp)


    vardep = testdata.iloc[:, -1]
    n = len(testdata)
    tmp = vardep.unique()
    depvalues = np.sort(tmp)



    if tree_algorithm == "CART":
        pandas2ri.activate()
        pred_prob = rpart.predict_rpart(model['tree'], newdata=testdata, type="prob")
        pred_class = pd.DataFrame(rpart.predict_rpart(model['tree'], newdata=testdata, type="class")).astype(int)

    pred_class2 = pred_class.astype(str)
    for i in range(1, len(depvalues) + 1):
      pred_class2[pred_class == i] = str(depvalues[i - 1])

    pred_class = pred_class2.values.flatten()
    del pred_class2
    ind = np.where(vardep != pred_class)[0]
    error_rate = len(ind) / n

    if output_pred is True:
      pred_all = pd.concat([vardep.reset_index(drop=True), pd.Series(pred_class, name="pred"), pd.DataFrame(pred_prob)], axis=1)
      pred_all.columns = ["dependent"] + ["pred"] + ["prob_" + str(i) for i in range(1, pred_prob.shape[1] + 1)]
    else:
      pred_all = None

    output = {'pred': pred_all, 'error_rate': error_rate}

    return output



def analyze_group_term_overlap(selected_coefficients):
    grpcnt = 0
    varnames = selected_coefficients.iloc[:, 1]  # Assuming the term names are in the first column
    for i in range(len(varnames)):
        if not varnames.iloc[i].startswith('rule'):
            # Check if the current term name appears in the names of other terms
            grpcnt += (sum(varnames.drop(i).str.contains(varnames.iloc[i])) > 0)
    return (grpcnt)

 




def plot_terms_SRE2(object, traindata):
    n_terms = len(object['regular_coefficients_selected'])# ya da .shape[0] kullanabilirsin
    regular_coefficients_selected = object['regular_coefficients_selected']
    robjects.globalenv['gam_model'] = object['splines']['gammodels'][0]    
    for j in range(n_terms):
        termtype = regular_coefficients_selected.iloc[j]['termtype']
        term = regular_coefficients_selected.iloc[j]['term']
        beta = regular_coefficients_selected.iloc[j]['beta']
        
        if termtype == "spline":
            # Spline terimi
            cleaned_term = term.replace('s(', '').replace(')', '')

            # R tarafında idx bulalım
            r.assign('cleaned_term', cleaned_term)
            idx_r = r('which(attr(gam_model$terms,"term.labels")==cleaned_term)')
            idx = int(np.array(idx_r)[0]) # R index 1-based, Python 0-based
            robjects.globalenv['idx'] = idx
            # R tarafında spline plot yap
            r('png("spline_plot.png")')
            r('plot(gam_model, select=idx)')
            r('dev.off()')
            from PIL import Image
            img = Image.open("spline_plot.png")
            img.show()
        
        elif termtype == "lterm":
            # Lineer terim
            
            # Python'dan eğitim verisinden x min-max al
            x_min = traindata[term].min()
            x_max = traindata[term].max()

            # Random x üret
            x = np.abs(np.random.uniform(low=x_min, high=x_max, size=20))
            y = beta * x

            # Çizim
            plt.figure()
            plt.plot(x, y, 'o')
            plt.plot([x_min, x_max], [beta*x_min, beta*x_max], linestyle='--', color='r')
            plt.xlabel(term)
            plt.ylabel('Effect')
            plt.title(f'Linear Term: {term}')
            plt.ylim([-0.6, 0.8])
            plt.grid(True)
            plt.show()

def plot_terms_SRE(object, traindata):
    for j in range(len(object['regular_coefficients_selected'])):
        termtype = object['regular_coefficients_selected'].iloc[j]['termtype']
        term = object['regular_coefficients_selected'].iloc[j]['term']
        beta = object['regular_coefficients_selected'].iloc[j]['beta']

        if termtype == "spline":
            # s() kaldır, sadece değişken ismini al
            cleaned_term = term.replace('s(', '').replace(')', '')
            
            # spline'lar arasında eşleşen indeks bulunuyor
            smooth_terms_list = object['splines']['smoothterms']['term']
            idx = [i for i, t in enumerate(smooth_terms_list) if t == cleaned_term]

            if idx:
                idx = idx[0]  # sadece ilk eşleşeni alıyoruz
                spline_model = object['splines']['smoothterms']['models'][idx]
                
                # spline modeline uygun x ve y alımı burada olmalı
                x = spline_model['x']  # veya generate edilecek x
                y = spline_model['y']  # spline çıktı tahminleri

                plt.figure()
                plt.plot(x, y)
                plt.xlabel(cleaned_term)
                plt.title(f"Spline Plot for {cleaned_term}")
                plt.show()

        elif termtype == "lterm":
            # lineer terimler için
            x_min = traindata[term].min()
            x_max = traindata[term].max()
            x = np.abs(np.random.uniform(low=x_min, high=x_max, size=20))
            y = beta * x

            plt.figure()
            plt.plot(x, y, 'o')
            plt.plot([x_min, x_max], [beta * x_min, beta * x_max], linestyle='--', color='r')
            plt.xlabel(term)
            plt.ylabel('y')
            plt.title(f"Linear Term Plot for {term}")
            plt.ylim([-0.6, 0.8])
            plt.show()

    
def plot_terms_SRE_combined(object, traindata):
    n_terms = len(object['regular_coefficients_selected'])
    regular_coefficients_selected = object['regular_coefficients_selected']
    robjects.globalenv['gam_model'] = object['splines']['gammodels'][0]

    fig, axes = plt.subplots(n_terms, 1, figsize=(8, 4 * n_terms))
    if n_terms == 1:
        axes = [axes]  # Tek terim varsa axes iterable değil, düzelt

    for j in range(n_terms):
        termtype = regular_coefficients_selected.iloc[j]['termtype']
        term = regular_coefficients_selected.iloc[j]['term']
        beta = regular_coefficients_selected.iloc[j]['beta']
        ax = axes[j]

        if termtype == "spline":
            # R tarafında spline plot'u temp dosyasına çiz
            cleaned_term = term.replace('s(', '').replace(')', '')
            r.assign('cleaned_term', cleaned_term)
            idx_r = r('which(attr(gam_model$terms,"term.labels")==cleaned_term)')
            idx = int(np.array(idx_r)[0])
            robjects.globalenv['idx'] = idx

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                tmpname = tmpfile.name
            r(f'png("{tmpname}")')
            r('plot(gam_model, select=idx)')
            r('dev.off()')

            # PNG'yi yükle ve matplotlib subplotuna çiz
            img = Image.open(tmpname)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Spline Term: {term}')
            os.remove(tmpname)

        elif termtype == "lterm":
            x_min = traindata[term].min()
            x_max = traindata[term].max()
            x = np.linspace(x_min, x_max, 100)
            y = beta * x

            ax.plot(x, y, linestyle='--', color='r')
            ax.set_xlabel(term)
            ax.set_ylabel('Effect')
            ax.set_title(f'Linear Term: {term}')
            ax.grid(True)
            ax.set_ylim([-0.6, 0.8])

    plt.tight_layout()
    output_file = "combined_terms_plot.png"
    plt.savefig(output_file)
    plt.close()
    
    img = Image.open(output_file)
    img.show() 

 

def plot_dependence_unique(object, traindata, datatype):
    """
    object: Eğitimli model
    traindata: Eğitim verisi
    datatype: "train"
    important_features: Feature listesi
    """
    important_features  = ["DELINQ", "NINQ", "DEROG", "DEROG.dummy", "JOB.binned_2", "VALUE.dummy", "CLAGE.dummy", "JOB.binned_1", "CLNO.dummy"] # Kendi önemli feature'larını buraya ekle
    results = []

    n_features = len(important_features)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows)
    )
    axes = axes.flatten()

    for idx, feature in enumerate(important_features):
        print(f"Partial Dependence hesaplanıyor: {feature}")

        feature_values = np.sort(traindata[feature].unique())
        mean_preds = []

        for val in feature_values:
            temp_data = traindata.copy()
            temp_data[feature] = val

            mean_pred = predict_SREMean(object, temp_data, datatype)
            mean_preds.append(mean_pred)

            results.append({
                'Feature': feature,
                'Feature_Value': val,
                'Predicted_Mean': mean_pred
            })

        # Tek bir panelde çiz
        ax = axes[idx]
        ax.plot(feature_values, mean_preds, linestyle='-')  # marker yok
        ax.plot(feature_values, mean_preds, linestyle='-')
        # Y eksenini genişlet
        ymin = min(mean_preds)
        ymax = max(mean_preds)
        padding = (ymax - ymin) * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence')
        ax.grid(False)

  

    # Eğer fazla boş kutu varsa onları kapatalım
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('partial_dependence_all.jpg', dpi=200)
    plt.close()

    print("\nTüm Partial Dependence grafikleri tek bir JPEG dosyasına kaydedildi!")
    
    pdp_df = pd.DataFrame(results)
    return pdp_df
def predict_SRE (object, newdata,datatype):
    depvalues = object['classes']
    robjects.globalenv['regular.model'] = object['regular_model']
    if object['add_rules']:
        n = len(newdata.iloc[:, 0])
        flat_list = [item for sublist in object['rules'] for item in sublist]
        ##ts = object['rules_rulemaptable'].shape[0]
        member_pred = pd.DataFrame(0, index=range(n), columns=range(len(flat_list)))
        ts = 0

        for m in range(object['rules_nrtrees']):
            if object['rules'][m] != "NONE":
                # Burada R'daki predict.rpart_rules fonksiyonunun Python versiyonunu kullanmalısınız
                pred_prob_m = predict_rpart_rules(object['rules'][m], newdata)
                nrr = pred_prob_m.shape[1]
                member_pred.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
                new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
                current_column_names = member_pred.columns.tolist()
                current_column_names[ts:ts+nrr] = new_column_names
                member_pred.columns = current_column_names
                ts += nrr

        rulepreds = member_pred
        rules_df=np.array(object['rules_rulemaptable']['term'])
        rulepreds = rulepreds[rules_df]
        ##rules_array = np.array(flat_list).reshape(-1, 1)
        # rules_array = np.array(object['rules_rulemaptable']['rule']).reshape(-1, 1)
        # rules_rulemaptable = pd.DataFrame(rules_array, columns=['V1'])
        # rules_rulemaptable = rules_rulemaptable[rules_rulemaptable != "NONE"]
        # colnames_df = pd.DataFrame(rulepreds.columns,columns=['V2'])
        # rules_rulemaptable = pd.concat([rules_rulemaptable, colnames_df], axis=1)
        # uniquerules = rules_rulemaptable.drop_duplicates('V1')['V2']
        # uniquerules = [re.sub(r"_.*", "", rule) for rule in uniquerules]
        # rulepreds = rulepreds[uniquerules]
        # rulepreds.columns = uniquerules



    if object['add_splines']:
        splinepreds = predict_gamsplines(object['splines'], newdata)


        splinepreds = 0.4 * (splinepreds / np.array(object['splines_stddevs']))

    if object['add_lterms']:
        ltermpreds = 0.4 * (newdata.iloc[:, :len(object['lterms_stddevs'])] / np.array(object['lterms_stddevs']))

        ##ltermpreds = newdata.iloc[:, :len(object['lterms_stddevs'])]

    if object['add_lterms'] and object['add_rules'] and object['add_splines']:
        termpredmatrix = pd.concat([rulepreds, splinepreds, ltermpreds,newdata.iloc[:, -1]], axis=1)
    elif object['add_lterms'] and object['add_rules']:
        termpredmatrix = pd.concat([rulepreds, ltermpreds], axis=1)
    elif object['add_rules']:
        termpredmatrix = rulepreds
    common_columns = object['reducedtermselmatrix'].columns.intersection(termpredmatrix.columns)
    last_col = termpredmatrix['dependent']
    
 
    
    
    termpredmatrix = termpredmatrix[common_columns]
    termpredmatrixcopy=termpredmatrix
    x_df= pd.DataFrame(termpredmatrix)
    x_df_copy= pd.DataFrame(termpredmatrixcopy)
    x_df_copy['dependent'] = last_col.values 
    
    if(datatype=="test"):
        x_df_copy.to_csv(f'{wd}/Data/Usmorttest_data.dat', sep=' ', index=False)
    else:
        x_df_copy.to_csv(f'{wd}/Data/Usmorttrain_data.dat', sep=' ', index=False)
    x_df =  x_df.iloc[:, :-1]
    termpredmatrix =  termpredmatrix.iloc[:, :-1]
    x = x_df.to_numpy()
    x=termpredmatrix.to_numpy()
     ##model = cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial',nfolds=cv_folds,alpha=regular_alpha,parallel=True,measure="AUC",)
    robjects.globalenv['x'] = numpy2ri.py2rpy(x)
    robjects.globalenv['regular.model'] = object['regular_model']
    if object['regular_type'] not in ["SGL2", "SGL","lasso2"]:
       robjects.r('''pred<-predict.glmnet(regular.model$glmnet.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
       pred = robjects.globalenv['pred']
    elif object['regular_type'] in ["SGL2", "lasso2"]:
       robjects.r('''pred<-predict(regular.model$oem.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
       pred = robjects.globalenv['pred']
    elif object['regular_type'] in ["SGL"]:
       robjects.r(''' todrop <- colnames(x) %in% droppedvariables
                      x <- x[,todrop]
                      pred<-predictSGL(regular.model,x,regular.model$idx)''')
       pred = robjects.globalenv['pred']


    pred_class = (pred > 0.5).astype(int) + 1
    depvalues_array = np.array(depvalues)
    pred_class = pd.Series(pred_class.flatten())
    pred_class_updated = pd.Series(depvalues_array[pred_class - 1], index=pred_class.index)

    pred_all = pd.DataFrame({'dependent': newdata.iloc[:, -1], 'pred_class': pred_class_updated, 'cl0': 1-pred.flatten(), 'cl1': pred.flatten(),'dependent_binary':(newdata.iloc[:, -1]== 'cl1').astype(int)})

    return {
        'pred': pred_all,
        'rule_preds': rulepreds,
        'ltermpreds': ltermpreds,
        'splinepreds': splinepreds
    }           
def predict_SREMean (object, newdata,datatype):
    depvalues = object['classes']
    robjects.globalenv['regular.model'] = object['regular_model']
    if object['add_rules']:
        n = len(newdata.iloc[:, 0])
        flat_list = [item for sublist in object['rules'] for item in sublist]
        ##ts = object['rules_rulemaptable'].shape[0]
        member_pred = pd.DataFrame(0, index=range(n), columns=range(len(flat_list)))
        ts = 0

        for m in range(object['rules_nrtrees']):
            if object['rules'][m] != "NONE":
                # Burada R'daki predict.rpart_rules fonksiyonunun Python versiyonunu kullanmalısınız
                pred_prob_m = predict_rpart_rules(object['rules'][m], newdata)
                nrr = pred_prob_m.shape[1]
                member_pred.iloc[:, ts:(ts+nrr)] = pred_prob_m.values
                new_column_names = ["rule{:05d}".format(i) for i in range(ts + 1, ts + nrr + 1)]
                current_column_names = member_pred.columns.tolist()
                current_column_names[ts:ts+nrr] = new_column_names
                member_pred.columns = current_column_names
                ts += nrr

        rulepreds = member_pred
        rules_df=np.array(object['rules_rulemaptable']['term'])
        rulepreds = rulepreds[rules_df]
        ##rules_array = np.array(flat_list).reshape(-1, 1)
        # rules_array = np.array(object['rules_rulemaptable']['rule']).reshape(-1, 1)
        # rules_rulemaptable = pd.DataFrame(rules_array, columns=['V1'])
        # rules_rulemaptable = rules_rulemaptable[rules_rulemaptable != "NONE"]
        # colnames_df = pd.DataFrame(rulepreds.columns,columns=['V2'])
        # rules_rulemaptable = pd.concat([rules_rulemaptable, colnames_df], axis=1)
        # uniquerules = rules_rulemaptable.drop_duplicates('V1')['V2']
        # uniquerules = [re.sub(r"_.*", "", rule) for rule in uniquerules]
        # rulepreds = rulepreds[uniquerules]
        # rulepreds.columns = uniquerules



    if object['add_splines']:
        splinepreds = predict_gamsplines(object['splines'], newdata)


        splinepreds = 0.4 * (splinepreds / np.array(object['splines_stddevs']))

    if object['add_lterms']:
        ltermpreds = 0.4 * (newdata.iloc[:, :len(object['lterms_stddevs'])] / np.array(object['lterms_stddevs']))

        ##ltermpreds = newdata.iloc[:, :len(object['lterms_stddevs'])]

    if object['add_lterms'] and object['add_rules'] and object['add_splines']:
        termpredmatrix = pd.concat([rulepreds, splinepreds, ltermpreds,newdata.iloc[:, -1]], axis=1)
    elif object['add_lterms'] and object['add_rules']:
        termpredmatrix = pd.concat([rulepreds, ltermpreds], axis=1)
    elif object['add_rules']:
        termpredmatrix = rulepreds
    common_columns = object['reducedtermselmatrix'].columns.intersection(termpredmatrix.columns)
    last_col = termpredmatrix['dependent']
    
 
    
    
    termpredmatrix = termpredmatrix[common_columns]
    termpredmatrixcopy=termpredmatrix
    x_df= pd.DataFrame(termpredmatrix)
    x_df_copy= pd.DataFrame(termpredmatrixcopy)
    x_df_copy['dependent'] = last_col.values 
    
    
    x_df =  x_df.iloc[:, :-1]
    termpredmatrix =  termpredmatrix.iloc[:, :-1]
    x = x_df.to_numpy()
    x=termpredmatrix.to_numpy()
     ##model = cvglmnet(x = x.copy(), y = y.copy(), family = 'binomial',nfolds=cv_folds,alpha=regular_alpha,parallel=True,measure="AUC",)
    robjects.globalenv['x'] = numpy2ri.py2rpy(x)
    robjects.globalenv['regular.model'] = object['regular_model']
    if object['regular_type'] not in ["SGL2", "SGL","lasso2"]:
       robjects.r('''pred<-predict.glmnet(regular.model$glmnet.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
       pred = robjects.globalenv['pred']
    elif object['regular_type'] in ["SGL2", "lasso2"]:
       robjects.r('''pred<-predict(regular.model$oem.fit,newx=x,type="response",s=regular.model$lambda.1se)''')
       pred = robjects.globalenv['pred']
    elif object['regular_type'] in ["SGL"]:
       robjects.r(''' todrop <- colnames(x) %in% droppedvariables
                      x <- x[,todrop]
                      pred<-predictSGL(regular.model,x,regular.model$idx)''')
       pred = robjects.globalenv['pred']


    
    #pred_all = pd.DataFrame({'dependent': newdata.iloc[:, -1], 'pred_class': pred_class_updated, 'cl0': 1-pred.flatten(), 'cl1': pred.flatten(),'dependent_binary':(newdata.iloc[:, -1]== 'cl1').astype(int)})

    return pred.flatten().mean()
       

def calculate_auc(predictions):
    df = pd.DataFrame(predictions)
    df['dependent_binary'] = (df['dependent'] == 'cl1').astype(int)
    # AUC hesaplamaları için gerekli tahmin değerleri ve gerçek etiketler
    predictions_cl1 = df['cl1']
    labels = df['dependent_binary']
    # cl0 ve cl1 için ayrı ayrı AUC hesapla
    return roc_auc_score(labels, predictions_cl1)


def calc_roc_curve(predictions):
    df = pd.DataFrame(predictions)
    df['dependent_binary'] = (df['dependent'] == 'cl1').astype(int)
    # AUC hesaplamaları için gerekli tahmin değerleri ve gerçek etiketler
    predictions_cl1 = df['cl1']
    labels = df['dependent_binary']
    # cl0 ve cl1 için ayrı ayrı AUC hesapla
    return roc_curve(labels, predictions_cl1)
# Read in training and test data
# df = pd.read_csv(f'{wd}/data/GMSC.csv')
# traindata, testdata = train_test_split(df, test_size=0.2, random_state=123)

# # Save mean and sd values
# traindata.to_csv(f'{wd}/data/GMSC_train.csv', index=False)
# testdata.to_csv(f'{wd}/data/GMSC_test.csv', index=False)

traindata = pd.read_csv(f'{wd}/Data/Usmort_train.csv')
testdata = pd.read_csv(f'{wd}/Data/Usmort_test.csv')



# Define feature subset for the case study (identified using an initial run)
#featureSet = ["callwait","changem","changem_M","creditde","custcare","directas","eqpdays","incalls","models","mou","mou_M","opeakvce","outcalls","phones","recchrge","retcalls","revenue","revenue_M","setprcm","webcap","dependent"]

# Take subset of features in training data set
traindata = traindata

# Take subset of features in training data set  (not used for case study)
testdata = testdata

#traindata['dependent'] = traindata['dependent'].map({'cl1': 'cl0', 'cl0': 'cl1'})
#testdata['dependent'] = testdata['dependent'].map({'cl1': 'cl0', 'cl0': 'cl1'})
# Data set descriptive statistics

# Obtain mean and sd values for all retained features, to be reported in the paper
traindata_means = traindata.iloc[:, :20].mean(numeric_only=True, skipna=True)
traindata_sd = traindata.iloc[:, :20].std(numeric_only=True, skipna=True)

# Mean values by class
traindata_means_cl0 = traindata[traindata["dependent"] == "cl0"].iloc[:, :20].mean(numeric_only=True, skipna=True)
traindata_means_cl1 = traindata[traindata["dependent"] == "cl1"].iloc[:, :20].mean(numeric_only=True, skipna=True)

import joblib

#termmod = SRE_termcreator(traindata, valdata=None, rules_nrtrees=10, maxdepth=10, splines_df="AUTO")
 
#joblib.dump(termmod, wd + "/Models/GMSCProfTreeRMSIndex.pickle")
model = joblib.load(wd+ "/Models/USMortProfTreeRMWithStability.pickle")

sremod2=SRE(traindata=traindata,
            termmodel=model,
            valdata=None,
            rules_nrtrees=100,
            regular_type="lasso2",
            regular_alpha=0.05,
            add_rules=True,
            add_splines=True,
            add_lterms=True,
            maxdepth=10,
            splines_df='AUTO',
            parallel=True,
            nr_cores=4,
            cv_metric="AUC",
            sgl_alpha=0.95,
            maxit=100,
            sgl_gamma=0.9,
            nlambda=100,
            sgl_lambdaselect="sparsestInsignDiffFromBest",
            sgl_reduce_ruleset=False,
            sgl_grouped_ruleset=False,
            cv_folds=10,
            standardize=True)


#joblib.dump(sremod2, wd + "/Models/GMSCProfTreeSRERMWithStability.pickle")


sremod2 = joblib.load(wd+ "/Models/UsmortSREMode.pickle")


SRE_SGL_preds_tr=predict_SRE(sremod2,traindata,"train")
SRE_SGL_preds_te=predict_SRE(sremod2,testdata,"test")


#resultDPD=plot_dependence_unique(sremod2, traindata, "train")
#plot_terms_SRE2(sremod2,traindata)
#plot_terms_SRE_combined(sremod2,traindata)
auc_train_1 = calculate_auc(SRE_SGL_preds_tr['pred'])
auc_test_1 = calculate_auc(SRE_SGL_preds_te['pred'])

print('auc train:')
print(auc_train_1 )
print('auc test:')
print(auc_test_1)
fpr_cl1, tpr_cl1, _ = calc_roc_curve(SRE_SGL_preds_te['pred'])

plt.figure()
#plt.plot(fpr_cl0, tpr_cl0, color='blue', lw=2, label=f'Class 0 (AUC = {auc_cl0:.2f})')
plt.plot(fpr_cl1, tpr_cl1, color='red', lw=2, label=f'Class 1 (AUC = {auc_test_1:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

score_ex_r = FloatVector(SRE_SGL_preds_te['pred']['cl1'].to_numpy())
class_ex_r = IntVector(SRE_SGL_preds_te['pred']['dependent_binary'].to_numpy())  # Gerçek sınıf etiketleri
result_default = emp.empChurn(score_ex_r, class_ex_r)
print("Standart EMP sonuçları:", result_default)
result_creditscore=emp.empCreditScoring(score_ex_r, class_ex_r)
print("Standart EMPCredit sonuçları:", result_creditscore)

# empcs(SRE_SGL_preds_te['pred']['dependent_binary'],SRE_SGL_preds_te['pred']['cl1'])

# mpcs_score(SRE_SGL_preds_te['pred']['dependent_binary'],SRE_SGL_preds_te['pred']['cl1'])
# ##empCreditScoring(SRE_SGL_preds_te['pred']['cl1'], SRE_SGL_preds_te['pred']['dependent_binary'])

# empcs_score(SRE_SGL_preds_te['pred']['dependent_binary'],SRE_SGL_preds_te['pred']['cl1'])

# empcs(SRE_SGL_preds_te['pred']['dependent_binary'],SRE_SGL_preds_te['pred']['cl1'])


