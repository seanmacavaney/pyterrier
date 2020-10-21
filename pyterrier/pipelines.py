from warnings import warn
from tqdm import tqdm
from collections import defaultdict
import itertools
import os
import pandas as pd
import numpy as np
from .utils import Utils
from .transformer import TransformerBase, EstimatorBase

def _bold_cols(data, col_type):
    if not data.name in col_type:
        return [''] * len(data)
    
    colormax_attr = f'font-weight: bold'
    colormaxlast_attr = f'font-weight: bold'
    if col_type[data.name] == "+":  
        max_value = data.max()
    else:
        max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

def _color_cols(data, col_type, 
                       colormax='antiquewhite', colormaxlast='lightgreen', 
                       colormin='antiquewhite', colorminlast='lightgreen' ):
    if not data.name in col_type:
      return [''] * len(data)
    
    if col_type[data.name] == "+":
      colormax_attr = f'background-color: {colormax}'
      colormaxlast_attr = f'background-color: {colormaxlast}'
      max_value = data.max()
    else:
      colormax_attr = f'background-color: {colormin}'
      colormaxlast_attr = f'background-color: {colorminlast}'
      max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

# The algorithm for grid search
def GridSearch(pipeline : TransformerBase, topics : pd.DataFrame, qrels : pd.DataFrame, param_map : dict, metric="ndcg", verbose=True):
    """
    Grid searches a set of named parameters on a given pipeline. The topics, qrels must be specified.
    The trec_eval measure name can be optionally specified.

    Transformers must be uniquely identified using the 'id' constructor kwarg. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of BatchRetrieve.

    Args:
        - pipeline(TransformerBase) : a retrieval system or a pipeline of such
        - topics(DataFrame): topics to tune upon
        - qrels(DataFrame): qrels to tune upon
        - param_map(dict): a two-level dictionary, mapping transformer id to param name to a list of values
        - metric(string): name of the metric to tune
        - verbose(bool): whether to use progress bars or not 

    *** Returns ***
        - A tuple containing the best transformer, and a two-level parameter map of the identified best settings

    :Example:

    >>> br = pt.BatchRetrieve(index, wmodel="BM25", controls={"c" : "0.25"}, id="bm25")
    >>> params = {
    >>>     'bm25' : {'c' : [b/10 for b in range(0,11)] }
    >>> }
    >>> rtr = pt.pipelines.GridSearchCV(br, tr_topics, qrels, params, metric="ndcg")
    >>> pt.Experiment([rtr], test_topics, qrels, eval_metrics=["ndcg"])
    """
    
    

    #Store the all parameter names and candidate values into a dictionary, keyed by a tuple of transformer id and parameter name
    #such as {('id1', 'wmodel'): ['BM25', 'PL2'], ('id1', 'c'): [0.1, 0.2, 0.3], ('id2', 'lr'): [0.001, 0.01, 0.1]}
    candi_dict = { (tran_id, param_name) : param_map[tran_id][param_name] for tran_id in param_map for param_name in param_map[tran_id]}
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")

    for tran_id in param_map:
        if pipeline.get_transformer(tran_id) is None:
            raise KeyError("No such transformer with id %s in retrieval pipeline %s" % (tran_id, str(pipeline)))

    # Iterate the candidate values in different combinations
    items = sorted(candi_dict.items())    
    keys,values = zip(*items)
    combinations = list(itertools.product(*values))
    
    eval_list = []
    #for each combination of parameter values
    for v in tqdm(combinations, total=len(combinations), desc="GridSearch", mininterval=0.3) if verbose else combinations:
        #'params' is every combination of candidates
        params = dict(zip(keys,v))
        parameter_list = []

        #Set the parameter value in the corresponding transformer of the pipeline
        for pipe_id, param_name in params:
            pipeline.get_transformer(pipe_id).set_parameter(param_name,params[pipe_id,param_name])
            # such as ('id1', 'wmodel', 'BM25')
            parameter_list.append((pipe_id,param_name,params[pipe_id,param_name]))
            
        # using topics and evaluation
        res = pipeline.transform(topics)
        eval_score = Utils.evaluate(res, qrels, metrics=[metric], perquery=False)[metric]
        # eval_list has the form [ ([('id1', 'wmodel', 'BM25'),('id1', 'c', 0.2),('id2', 'lr', '0.001')],0.2654)], where 0.2654 is the evaluation score.
        eval_list.append( (parameter_list, eval_score) )


    # identify the best setting
    best_score = 0
    max_index = 0
    for i, (param_list, score) in enumerate(eval_list):
        if score > best_score:
            best_score = score
            max_index = i
    best_params, _ = eval_list[max_index]
    
    best_params_map = { tran_id : {} for tran_id in param_map }
    for tran_id, param_name, param_value in best_params:
        best_params_map[tran_id][param_name] = param_value

    # configure the pipeline
    for i in range(len(best_params)):
        if not hasattr(pipeline,"id"):
            pipeline.get_transformer(best_params[i][0]).set_parameter(best_params[i][1],best_params[i][2])
        else:
            pipeline.set_parameter(best_params[i][1],best_params[i][2])
    best_transformer = pipeline

    # display the best results
    if verbose:
        #print best evaluation results
        print("The best %s score is: %f" % (metric, best_score))
        #print the best param map  
        print("The best parameters map is :")
        for i in range(len(best_params)):
            print(best_params[i])

    return best_transformer, best_params_map
    
# The algorithm for grid search with cross validation
def GridSearchCV(pipeline : TransformerBase, topics : pd.DataFrame, qrels : pd.DataFrame, param_map : dict, metric='ndcg', num_folds=5, **kwargs):
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    all_split_scores={}
    all_params=[]

    for train_index, test_index in KFold(n_splits=num_folds).split(topics):
        topics_train, topics_test = topics.iloc[train_index],topics.iloc[test_index]
        best_transformer, params = GridSearch(pipeline, topics_train, qrels, param_map, metric=metric)
        all_params.append(params)

        test_res = best_transformer.transform(topics_test)
        test_eval = Utils.evaluate(test_res, qrels, metrics=[metric], perquery=True)
        all_split_scores.update(test_eval)
    return all_split_scores, all_params

def Experiment(retr_systems, topics, qrels, eval_metrics, names=None, perquery=False, dataframe=True, baseline=None, highlight=None):
    """
    Cornac style experiment. Combines retrieval and evaluation.
    Allows easy comparison of multiple retrieval systems with different properties and controls.

    Args:
        retr_systems(list): A list of BatchRetrieve objects to compare
        topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
        qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'label']   
        eval_metrics(list): Which evaluation metrics to use. E.g. ['map']
        names(list)=List of names for each retrieval system when presenting the results.
            Default=None. If None: Use names of weighting models for each retrieval system.
        perquery(bool): If true return each metric for each query, else return mean metrics. Default=False.
        dataframe(bool): If True return results as a dataframe. Else as a dictionary of dictionaries. Default=True.
        baseline(int): If set to the index of an item of the retr_system list, will calculate the number of queries improved, degraded and the statistical significance (paired t-test p value) for each measure.
            Default=None: If None, no additional columns added for each measure
        highlight(str) : If "bold", highlights in bold the best measure value in each column; 
            if "color" or "colour" uses green to indicate highest values

    Returns:
        A Dataframe with each retrieval system with each metric evaluated.
    """
    
    
    # map to the old signature of Experiment
    warn_old_sig=False
    if isinstance(retr_systems, pd.DataFrame) and isinstance(topics, list):
        tmp = topics
        topics = retr_systems
        retr_systems = tmp
        warn_old_sig = True
    if isinstance(eval_metrics, pd.DataFrame) and isinstance(qrels, list):
        tmp = eval_metrics
        eval_metrics = qrels
        qrels = tmp
        warn_old_sig = True
    if warn_old_sig:
        warn("Signature of Experiment() is now (retr_systems, topics, qrels, eval_metrics), please update your code", DeprecationWarning, 2)
    
    if baseline is not None:
        assert int(baseline) >= 0 and int(baseline) < len(retr_systems)
        assert not perquery

    if isinstance(topics, str):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if isinstance(qrels, str):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)

    results = []
    neednames = names is None
    if neednames:
        names = []
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")
    for system in retr_systems:
        results.append(system.transform(topics))
        if neednames:
            names.append(str(system))

    qrels_dict = Utils.convert_qrels_to_dict(qrels)
    all_qids = topics["qid"].values

    evalsRows=[]
    evalDict={}
    evalDictsPerQ=[]
    actual_metric_names=[]
    for name,res in zip(names,results):
        evalMeasuresDict = Utils.evaluate(res, qrels_dict, metrics=eval_metrics, perquery=perquery or baseline is not None)
        
        if perquery or baseline is not None:
            # this ensures that all queries are present in various dictionaries
            # its equivalent to "trec_eval -c"
            (evalMeasuresDict, missing) = Utils.ensure(evalMeasuresDict, eval_metrics, all_qids)
            if missing > 0:
                warn("%s was missing %d queries, expected %d" % (name, missing, len(all_qids) ))

        if baseline is not None:
            evalDictsPerQ.append(evalMeasuresDict)
            evalMeasuresDict = Utils.mean_of_measures(evalMeasuresDict)

        if perquery:
            for qid in all_qids:
                for measurename in evalMeasuresDict[qid]:
                    evalsRows.append([name, qid, measurename,  evalMeasuresDict[qid][measurename]])
            evalDict[name] = evalMeasuresDict
        else:
            actual_metric_names = list(evalMeasuresDict.keys())
            evalMeasures = [evalMeasuresDict[m] for m in actual_metric_names]
            evalsRows.append([name]+evalMeasures)
            evalDict[name] = evalMeasures
    if dataframe:
        if perquery:
            return pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"])

        highlight_cols = { m : "+"  for m in actual_metric_names }

        if baseline is not None:
            assert len(evalDictsPerQ) == len(retr_systems)
            from scipy import stats
            baselinePerQuery={}
            for m in actual_metric_names:
                baselinePerQuery[m] = np.array([ evalDictsPerQ[baseline][q][m] for q in evalDictsPerQ[baseline] ])

            for i in range(0, len(retr_systems)):
                additionals=[]
                if i == baseline:
                    additionals = [None] * (3*len(actual_metric_names))
                else:
                    for m in actual_metric_names:
                        # we iterate through queries based on the baseline, in case run has different order
                        perQuery = np.array( [ evalDictsPerQ[i][q][m] for q in evalDictsPerQ[baseline] ])
                        delta_plus = (perQuery > baselinePerQuery[m]).sum()
                        delta_minus = (perQuery < baselinePerQuery[m]).sum()
                        p = stats.ttest_rel(perQuery, baselinePerQuery[m])[1]
                        additionals.extend([delta_plus, delta_minus, p])
                evalsRows[i].extend(additionals)
            delta_names=[]
            for m in actual_metric_names:
                delta_names.append("%s +" % m)
                highlight_cols["%s +" % m] = "+"
                delta_names.append("%s -" % m)
                highlight_cols["%s -" % m] = "-"
                delta_names.append("%s p-value" % m)
            actual_metric_names.extend(delta_names)

        df = pd.DataFrame(evalsRows, columns=["name"] + actual_metric_names)
        
        if highlight == "color" or highlight == "colour" :
            df = df.style.apply(_color_cols, axis=0, col_type=highlight_cols)
        elif highlight == "bold":
            df = df.style.apply(_bold_cols, axis=0, col_type=highlight_cols)
            
        return df 
    return evalDict


class LTR_pipeline(EstimatorBase):
    """
    This class simplifies the use of Scikit-learn's techniques for learning-to-rank.
    """
    def __init__(self, LTR, *args, fit_kwargs={}, **kwargs):
        """
        Init method

        Args:
            LTR: The model which to use for learning-to-rank. Must have a fit() and predict() methods.
            fit_kwargs: A dictionary containing additional arguments that can be passed to LTR's fit() method.  
        """
        self.fit_kwargs = fit_kwargs
        super().__init__(*args, **kwargs)
        self.LTR = LTR

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid=None, qrelsValid=None):
        """
        Trains the model with the given topics.

        Args:
            topicsTrain(DataFrame): A dataframe with the topics to train the model
        """
        if len(topics_and_results_Train) == 0:
            raise ValueError("No topics to fit to")
        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved")
        train_DF = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        kwargs = self.fit_kwargs
        self.LTR.fit(np.stack(train_DF["features"].values), train_DF["label"].values, **kwargs)
        return self

    def transform(self, test_DF):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

class XGBoostLTR_pipeline(LTR_pipeline):
    """
    This class simplifies the use of XGBoost's techniques for learning-to-rank.
    """

    def transform(self, topics_and_docs_Test):
        """
        Predicts the scores for the given topics.

        Args:
            topicsTest(DataFrame): A dataframe with the test topics.
        """
        test_DF = topics_and_docs_Test
        # xgb is more sensitive about the type of the values.
        test_DF["score"] = self.LTR.predict(np.stack(test_DF["features"].values))
        return test_DF

    def fit(self, topics_and_results_Train, qrelsTrain, topics_and_results_Valid, qrelsValid):
        """
        Trains the model with the given training and validation topics.

        Args:
            topics_and_results_Train(DataFrame): A dataframe with the topics and results to train the model
            topics_and_results_Valid(DataFrame): A dataframe with the topics and results for validation
        """
        if len(topics_and_results_Train) == 0:
            raise ValueError("No training results to fit to")
        if len(topics_and_results_Valid) == 0:
            raise ValueError("No validation results to fit to")

        if 'features' not in topics_and_results_Train.columns:
            raise ValueError("No features column retrieved in training")
        if 'features' not in topics_and_results_Valid.columns:
            raise ValueError("No features column retrieved in validation")

        tr_res = topics_and_results_Train.merge(qrelsTrain, on=['qid', 'docno'], how='left').fillna(0)
        va_res = topics_and_results_Valid.merge(qrelsValid, on=['qid', 'docno'], how='left').fillna(0)

        kwargs = self.fit_kwargs
        self.LTR.fit(
            np.stack(tr_res["features"].values), tr_res["label"].values, 
            group=tr_res.groupby(["qid"]).count()["docno"].values, # we name group here for libghtgbm compat. 
            eval_set=[(np.stack(va_res["features"].values), va_res["label"].values)],
            eval_group=[va_res.groupby(["qid"]).count()["docno"].values],
            **kwargs
        )

class PerQueryMaxMinScoreTransformer(TransformerBase):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
