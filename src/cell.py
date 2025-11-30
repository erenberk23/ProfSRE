 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import _tree
import rpy2
import os
import ctypes
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.3"
# Load DLL into memory.
hllDll = ctypes.WinDLL (r"C:\Program Files\R\R-4.2.3\bin\x64\R.dll")
from sklearn.metrics import confusion_matrix
from rpy2.robjects import pandas2ri, packages
import matplotlib.pyplot as plt
pandas2ri.activate()
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
from pygam import LinearGAM, LogisticGAM
from pygam import LogisticGAM, s, te 
wd = "C:/Users/Eren/Desktop/Cell2Cell/"
os.chdir(wd)
 
 
 

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
        
        # Build decision tree classifier
        if minbucketsize != "AUTO":
            fit = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth, 
                                         min_samples_leaf=minbucketsize, class_weight=caseweights)
        else:
            fit = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth, class_weight=caseweights)
        
        # Perform pruning if specified
        if pruning:
            fit = fit.fit(traindata.iloc[:,:-1], traindata[dependent])
            fit = DecisionTreeClassifier(criterion="entropy", max_depth=maxdepth, 
                                          min_samples_leaf=minbucketsize, ccp_alpha=cp, 
                                          class_weight=caseweights).fit(traindata.iloc[:,:-1], traindata[dependent])
        else:
            fit = fit.fit(traindata.iloc[:,:-1], traindata[dependent])
        
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

def plot_decision_tree(model, feature_names, class_names):
    # plot_tree function contains a list of all nodes and leaves of the Decision tree
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)
    
    # I return the tree for the next part
    return tree
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
        tree_to_code(dt,dt.feature_names_in_)
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

    if add_rules:
        for m in range(rules_nrtrees):
            print(f"training tree {m + 1}")
            bootstrap = np.random.choice(range(1, n),n, replace=True,)
            traindata_m = traindata.iloc[bootstrap]
            member = tree(traindata_m,tree_algorithm="CART",pruning=False,output_pred=False,maxdepth=10)
            tmp = pd.DataFrame(predict_tree(member,traindata), columns=[f"V{nclasses + i + 1}" for i in range(nclasses - 1)])
            pred_prob_m = tmp.iloc[:, 2:]
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
                nrr = pred_prob_m.shape[1] - 2
                cnt += 1
                member_pred.iloc[:, ts:ts + nrr] = pred_prob_m.iloc[:, 2:]
                member_pred.columns[ts:ts + nrr] = [f"rule{str(i + ts + 1).zfill(5)}" for i in range(nrr)]
                ts += nrr

        rulepreds = member_pred
        rules_support = rulepreds.mean()
        rules_scale = np.sqrt(rulepreds.mean() * (1 - rulepreds.mean()))
        rules_rulemaptable = np.concatenate((rules, np.array(rulepreds.columns)[:, np.newaxis], rules_support[:, np.newaxis], rules_scale[:, np.newaxis]), axis=1)
        rules_rulemaptable = rules_rulemaptable[rules_rulemaptable[:, 0] != "NONE", :]
        uniquerules = np.unique(rules_rulemaptable[:, 1])
        rulepreds = rulepreds[uniquerules]
        rules_rulemaptable = pd.DataFrame(rules_rulemaptable, columns=["rule", "term", "support", "scale"])

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
            splines_stddevs[j] = np.std(splinepreds[:, j])
            if splines_stddevs[j] == 0:
                splines_stddevs[j] = 1
        splines_valdata = 0.4 * np.divide(splinepreds[:, :splinepreds.shape[1]], splines_stddevs.T)
    else:
        splines = None
        splines_stddevs = None

    if add_lterms:
        lterms_stddevs = np.zeros((p, 1))
        for j in range(p):
            lterms_stddevs[j] = np.std(traindata[:, j])
            if lterms_stddevs[j] == 0:
                lterms_stddevs[j] = 1
        lterms_valdata = 0.4 * np.divide(valdata[:, :p], lterms_stddevs.T)
        lterms = traindata.columns[:p]
    else:
        lterms_stddevs = None
        lterms = None

    if add_lterms and add_rules and add_splines:
        termselmatrix = np.concatenate((rulepreds, splines_valdata, lterms_valdata, valdata[:, -1].reshape(-1, 1)), axis=1)
    elif add_lterms and add_rules:
        termselmatrix = np.concatenate((rulepreds, lterms_valdata, valdata[:, -1].reshape(-1, 1)), axis=1)
    elif add_rules:
        termselmatrix = np.concatenate((rulepreds, valdata[:, -1].reshape(-1, 1)), axis=1)
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
       'rules.nrtrees': rules.nrtrees,
       'classes': depvalues,
       'termselmatrix': termselmatrix,
       'terms.rules': rulepreds,
       'terms.splines': splines_valdata,
       'terms.linear': lterms_valdata}

    ans = dict(ans)
    ans['type'] = "SREterms"

 

 

def gamsplines(traindata, df="AUTO", gamtype="gam", threshold=5, splinetype=["tp", "cr", "cs", "ps", "ts"]):
    # Binary classification only!!!
    formula = traindata.columns[0] + " ~ " + " + ".join(traindata.columns[1:])
    cutoff = 0.5
    selection = False
    vardep = traindata.iloc[:, 0]
    depvalues = np.unique(vardep)
    n = traindata.shape[0]
    p = traindata.shape[1] - 1
    maxdepth = len(np.unique(vardep))
    nclasses = len(depvalues)
    gammodels = []
    for j in range(1, (p - 1) // 10 + 2):
        cols = [x for x in traindata.columns[(10*(j-1)+1):min(p,10*j)] if x != formula.split()[0]]
        traindata_s = traindata[[formula.split()[0]] + cols]
        rfs_linvars = []
        rfs_nparvars = []
        for vn in cols:
            uniq = traindata_s[vn].nunique()
            if 1 < uniq <= threshold:
                rfs_linvars.append(vn)
            elif uniq > threshold:
                rfs_nparvars.append(vn)
        if len(rfs_nparvars) > 0:
            if df != "AUTO":
                npar_form1 = [s(x, bs=splinetype, k=df+1, fx=True) for x in rfs_nparvars]
                npar_form2 = te(*npar_form1).terms
            if df == "AUTO":
                npar_form1 = [s(x, bs=splinetype) for x in rfs_nparvars]
                npar_form2 = te(*npar_form1).terms
        else:
            npar_form2 = []
        if len(rfs_linvars) > 0:
            lin_form = " + ".join(rfs_linvars)
            if len(rfs_nparvars) > 0:
                finalstring = formula.split()[0] + " ~ " + " + ".join(npar_form2) + " + " + lin_form
            else:
                finalstring = formula.split()[0] + " ~ " + lin_form
        else:
            finalstring = formula.split()[0] + " ~ " + " + ".join(npar_form2)
        if gamtype == "bam":
            if df != "AUTO":
                gammodels.append(LogisticGAM.from_formula(finalstring, data=traindata_s, family='binomial', link='logit', select=selection))
            else:
                gammodels.append(LogisticGAM.from_formula(finalstring, data=traindata_s, family='binomial', link='logit', select=selection, method='GCV.Cp'))
        elif gamtype == "gam":
            if df != "AUTO":
                gammodels.append(LogisticGAM.from_formula(finalstring, data=traindata_s, family='binomial', link='logit', select=selection))
            else:
                gammodels.append(LogisticGAM.from_formula(finalstring, data=traindata_s, family='binomial', link='logit', select=selection, method='GCV.Cp'))
        if j == 1:
                smoothterms = np.ravel(rfs_nparvars)
                modelids = np.repeat(j, len(rfs_nparvars))
                splineids = np.arange(1, len(rfs_nparvars) + 1)
        else:
                smoothterms = np.concatenate((smoothterms, np.ravel(rfs_nparvars)))
                modelids = np.concatenate((modelids, np.repeat(j, len(rfs_nparvars))))
                splineids = np.concatenate((splineids, np.arange(1, len(rfs_nparvars) + 1)))

    smoothterms = np.column_stack((smoothterms, modelids, splineids, ['s({})'.format(term) for term in smoothterms]))
    colnames = ["var", "modelid", "splineid", "term"]
    ans = {"gammodels": gammodels, "nmodels": j, "smoothterms": smoothterms}

def extract_rpart_rules(model, type=["probs","binary"]):
    if model.splits is None:
        ruleset = "NONE"
    else:
        relsplits = model.splits[model.splits.index.isin(model.frame['var'])]
        var = relsplits.index.tolist()
        relsplits = relsplits.reset_index(drop=True)
        relsplits.insert(0, 'var', var)
        frame = pd.DataFrame(model.frame).reset_index().rename(columns={'index':'node'})
        frame['node'] = frame['node'].astype(int)
        frame = frame.sort_values('node').reset_index(drop=True)
        splitvars = frame['var'][~frame['var'].isin(['<leaf>'])]
        splitvars_n = frame.loc[~frame.duplicated(subset='var', keep='last') & (frame['var'] != '<leaf>'), 'n']
        addcol1 = pd.concat([pd.DataFrame(['root']), pd.DataFrame(np.repeat(splitvars, 2))]).reset_index(drop=True)
        addcol2 = pd.concat([pd.DataFrame([max(frame['n'])]), pd.DataFrame(np.repeat(splitvars_n, 2))]).reset_index(drop=True)
        addcol1.columns = ['splitvar']
        addcol2.columns = ['n2']
        frame = pd.concat([frame, addcol1, addcol2], axis=1)
        frame = frame.rename(columns={'var':'oldvar', 'splitvar':'var'})
        rules = pd.merge(frame, relsplits, on='var')
        rules = rules[rules['n2'] == rules['count']]
        rules.loc[~rules.duplicated(subset='node', keep='last'), 'ncat'] *= -1
        operator = np.where(rules['ncat'] == 1, '>=', '<')
        rules.insert(4, 'operator', operator)
        rules = rules[['node', 'var', 'operator', 'index', 'n', 'yval2']]
        rules[['p_cl1', 'p_cl0']] = pd.DataFrame(rules.pop('yval2').tolist(), index=rules.index)
        rules = rules.sort_values('node', ascending=False).reset_index(drop=True)
        parents = np.zeros((rules.shape[0], 1))
        for i in range(0, rules.shape[0], 2):
            s = rules.loc[i, 'n'] + rules.loc[i+1, 'n']
            parent = rules.loc[rules['n'] == s, 'node'].tolist()
            if len(parent) > 0:
                parents[i] = parent[0]
                parents[i+1] = parent[0]
        rules['parents'] = parents
        rules = rules.sort_values('node', ascending=False).reset_index(drop=True)
        ruleset = []
        for i in range(rules.shape[0]):
            for j in range(i, rules.shape[0]):
                if j == i:
                    newrule = f"np.where(((indata['{rules.loc[j, 'var']}'] {rules.loc[j, 'operator']} {rules.loc[j, 'index']}) "
                    parent_idx = rules.loc[j, 'parents']
                    if type == 'probs':
                        p_cl1 = rules.loc[j, 'p_cl1']
                    elif type == 'binary':
                        p_cl1 = 1
                if rules.iloc[j]['node'] == parent_idx:
                    newrule += f"*(indata${rules.iloc[j]['var']} {rules.iloc[j]['operator']} {rules.iloc[j]['threshold']})"
                    parent_idx = rules.iloc[j]['parents']
            newrule = f"{newrule}), {p_cl1}, {1-p_cl1})"
            ruleset.append(newrule)
    return ruleset


def predict_rpart_rules(ruleset, indata, predvarprefix=None):
    predictions = np.zeros((indata.shape[0], len(ruleset)))
    for i in range(len(ruleset)):
        predictions[:, i] = eval(ruleset[i])
    predictions = pd.DataFrame(predictions)
    if predvarprefix is not None:
        colnames = [f"{predvarprefix}_r{i+1}" for i in range(len(ruleset))]
        predictions.columns = colnames
    return predictions


def predict_gamsplines(model, data):
    for j in range(model["nmodels"]):
        pred = predict_gam(model["gammodels"][j], data, type="terms")
        pred = pd.DataFrame(pred)
        if j == 0:
            predictions = pred
        else:
            predictions = pd.concat([predictions, pred], axis=1)
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

# Read in training and test data
traindata = pd.read_csv(f'{wd}/data/ds5_cv1tr1_u.csv')
testdata = pd.read_csv(f'{wd}/data/ds5_cv1te1.csv')

# Define feature subset for the case study (identified using an initial run)
featureSet = ["callwait","changem","changem_M","creditde","custcare","directas","eqpdays","incalls","models","mou","mou_M","opeakvce","outcalls","phones","recchrge","retcalls","revenue","revenue_M","setprcm","webcap","dependent"]

# Take subset of features in training data set 
traindata = traindata[featureSet]

# Take subset of features in training data set  (not used for case study)
testdata = testdata[featureSet]

# Data set descriptive statistics

# Obtain mean and sd values for all retained features, to be reported in the paper
traindata_means = traindata.iloc[:, :20].mean(numeric_only=True, skipna=True)
traindata_sd = traindata.iloc[:, :20].std(numeric_only=True, skipna=True)

# Mean values by class
traindata_means_cl0 = traindata[traindata["dependent"] == "cl0"].iloc[:, :20].mean(numeric_only=True, skipna=True)
traindata_means_cl1 = traindata[traindata["dependent"] == "cl1"].iloc[:, :20].mean(numeric_only=True, skipna=True)

# Save mean and sd values
traindata_means.to_csv(f'{wd}/Descriptives/cell2cell_traindata_means.csv', index=False)
traindata_sd.to_csv(f'{wd}/Descriptives/cell2cell_traindata_sd.csv', index=False)


termmod = SRE_termcreator(traindata, valdata=None, rules_nrtrees=10, maxdepth=10, splines_df="AUTO")
import pickle

with open(wd + "/Models/cell2cell_termmod.pickle", "wb") as file:
    pickle.dump(termmod, file)
