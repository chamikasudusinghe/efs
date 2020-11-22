import warnings
import random
from copy import deepcopy
import math
import itertools
import numpy as np
import collections

from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from cela.statistics import Statistics
from cela.feature import Feature
from cela.range_operation import RangeOperation
from cela.operator import default_operators


REGRESSION = 'regression'
CLASSIFICATION = 'classification'
ZERO_OUT = 'zero_out'
COEFFICIENT_RANK = 'coefficient_rank'


def build_operation_stack(string):
    stack = []
    start = 0
    for i, s in enumerate(string):
        if s == '(':
            substring = string[start:i]
            start = i + 1
            stack.append(substring)
        elif s == ',':
            if i != start:
                substring = string[start:i]
                stack.append(substring)
            start = i + 1
        elif s == ')':
            if i != start:
                substring = string[start:i]
                stack.append(substring)
            start = i + 1
    return stack


def get_model_string(features):
    feature_strings = []
    for f in features:
        feature_strings.append(f.string)
    return '[' + '] + ['.join(feature_strings) + ']'


def get_feature_value(stack, feature_names, X, variable_type_indices, operators):
    variables_stack = []
    while len(stack) > 0:
        current = stack.pop()
        if variable_type_indices and current.startswith('RangeOperation'):
            range_operation = RangeOperation(variable_type_indices, feature_names, X, string=current)
            variables_stack.append(np.squeeze(range_operation.value))
        elif current in feature_names:
            variable_index = feature_names.index(current)
            variables_stack.append(X[:, variable_index])
        elif operators.contains(current):
            operator = operators.get(current)
            variables = []
            for _ in range(operator.parity):
                variables.append(variables_stack.pop())
            result = operator.operation(*variables)
            variables_stack.append(result)
    return variables_stack.pop()


def build_basis_from_features(infix_features, predictor_names, X, variable_type_indices, operators):
    basis = np.zeros((X.shape[0], len(infix_features)))
    for j, f in enumerate(infix_features):
        if variable_type_indices and f.startswith('RangeOperation'):
            range_operation = RangeOperation(variable_type_indices, predictor_names, X, string=f)
            basis[:, j] = np.squeeze(range_operation.value)
        elif f in predictor_names:
            variable_index = predictor_names.index(f)
            basis[:, j] = X[:, variable_index]
        else:
            operation_stack = build_operation_stack(f)
            basis[:, j] = get_feature_value(operation_stack, predictor_names, X, variable_type_indices, operators)
    return basis


def get_basis_from_infix_features(infix_features, feature_names, X, scaler=None, variable_type_indices=None,
                                  operators=default_operators):
    basis = build_basis_from_features(infix_features, feature_names, X, variable_type_indices, operators)
    basis = np.nan_to_num(basis)
    if scaler:
        basis = scaler.transform(basis)
    return basis


class EvolutionaryFeatureSynthesis:

    def __init__(self, seed=None, fitness_algorithm=COEFFICIENT_RANK, method=None, max_gens=10,
                 num_additions=None, normalize=True, preserve_originals=True, tournament_probability=.9,
                 max_useless_steps=200, fitness_threshold=.01, correlation_threshold=0.95, reinit_range_operators=3,
                 splits=3, time_series_cv=False, range_operators=0, variable_type_indices=None,
                 operators=default_operators, verbose=1):
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(100000)
        self.fitness_algorithm = fitness_algorithm
        self.method = method
        self.max_gens = max_gens
        self.num_additions = num_additions
        self.normalize = normalize
        self.preserve_originals = preserve_originals
        self.tournament_probability = tournament_probability
        self.max_useless_steps = max_useless_steps
        self.fitness_threshold = fitness_threshold
        self.correlation_threshold = correlation_threshold
        self.reinit_range_operators = reinit_range_operators
        self.splits = splits
        self.time_series_cv = time_series_cv
        self.range_operators = range_operators
        self.variable_type_indices = variable_type_indices
        self.operators = operators
        self.verbose = verbose

    def fit(self, X, y, feature_names=None):
        self.models_ = []
        self.features_ = []
        self.scalers_ = []
        self.validation_scores_ = []
        self.statistics_ = Statistics()
        self.response_variance_ = np.var(y)
        self.predictor_names_, self.current_features_ = self._init_features(feature_names, X)
        #if self.verbose >= 2:
            #print('Starting modeling process with features: ')
            #for f in self.current_features_:
            #    print(f.string)
        self._init_others(X)
        best_score = 10000000
        steps_without_new_model = 0
        gen = 1
        best_gen = 1
        while gen <= self.max_gens and steps_without_new_model <= self.max_useless_steps:
            if self.verbose >= 2:
                print('Generation: ' + str(gen))
            score, model, scaler, y_pred = self._score_model(y)
            self.statistics_.add(gen, score, len(y))
            if self.verbose >= 2:
                current_features_elements_generation = []
                for features_gen in range (len(self.current_features_)):
                    current_features_elements_generation.append(self.current_features_[features_gen].string)
                print('Current Features: ' + get_model_string(self.current_features_) + ' Count:',str(len(list(set(current_features_elements_generation)))))
            if self.verbose >= 2:
                print('Score: ' + str(score))
                if (best_score != 10000000):
                    print('Best Model Score: ' + str(best_score) + ' Generation Number :' +str(best_gen))
            if score < best_score:
                self.validation_scores_.append(score)
                steps_without_new_model = 0
                best_score = score
                best_gen = gen
                if self.verbose >= 1:
                    print('New Best Model Score: ' + str(best_score))
                self.models_.append(model)
                self.scalers_.append(scaler)
                temp_features = deepcopy(self.current_features_)
                for f in temp_features:
                    f.value = None
                self.features_.append(temp_features)
            else:
                steps_without_new_model += 1
            if self.verbose >= 1:
                #if gen>1:
                    #print("Best Model:", self.models_[-1])
                    #print("Best Scalars:",self.scalers_[-1])
                #    print(self.current_features_ cc  cx )
                print('-------------------------------------------------------')
            if gen < self.max_gens and steps_without_new_model <= self.max_useless_steps:
                self._compose_features()     
                self._update_fitness(y)
                if self.verbose >= 2:
                    print('Top performing features:')
                    for i in range(3 if len(self.current_features_) >= 3 else len(self.current_features_)):
                        print(self.current_features_[i].string + ': ' + str(self.current_features_[i].fitness))
                if gen % self.reinit_range_operators == 0:
                    self._swap_range_operators(X)
            gen += 1
        self.best_model_= self.models_[-1]
        self.best_scaler_ = self.scalers_[-1]
        self.best_features_ = self.features_[-1]

        return self.best_features_,best_gen

    def predict(self, X):
        basis = self._get_basis_for_predictors(X)
        return self.best_model_.predict(basis)

    def _get_basis_for_predictors(self, X):
        infix_features = list(map(lambda x: x.infix_string, self.best_features_))
        basis = get_basis_from_infix_features(infix_features, self.predictor_names_, X, self.best_scaler_,
                                              self.variable_type_indices, self.operators)
        return basis

    def score(self, X, y):
        basis = self._get_basis_for_predictors(X)
        y_pred = self.best_model_.predict(basis)
        return mean_squared_error(y,y_pred)

    def pred(self, X, y):
        basis = self._get_basis_for_predictors(X)
        y_pred = self.best_model_.predict(basis)
        return y_pred

    def _init_others(self, X):
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.num_additions is None:
            self.num_additions = math.ceil(X.shape[1] / 3)

    def _init_features(self, feature_names, X):
        """
        Init the features the model will begin evolving. If feature names have not been specified they are
        generated as x1, x2, ..., xk.

        Parameters
        ----------
        feature_names : list of strings
            Names of the features.

        X : ndarray
            training data

        """

        features = []
        if feature_names is None:
            feature_names = ['x' + str(x) for x in range(X.shape[1])]
        for i, name in enumerate(feature_names):
            features.append(Feature(X[:, i], name, name, original_variable=self.preserve_originals))
        for _ in range(self.range_operators):
            features.append(RangeOperation(self.variable_type_indices, feature_names, X))
        return feature_names, features

    def _get_current_basis(self):
        basis = np.zeros((self.current_features_[0].value.shape[0], len(self.current_features_)))
        for i, f in enumerate(self.current_features_):
            basis[:, i] = self.current_features_[i].value
        basis = np.nan_to_num(basis)
        scaler = None
        if self.normalize:
            scaler = StandardScaler()
            basis = scaler.fit_transform(basis)
        return basis, scaler

    def _check_equal(self,items):
        return all(x == items[0] for x in items)

    def _build_linear_model(self, basis, y):
        if self.time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.splits)
        else:
            cv = KFold(n_splits=self.splits, random_state=self.seed, shuffle = True)
        if self.method == REGRESSION:
            #model = XGBRegressor(objective='reg:squarederror',booster='gbtree')
            #model = ElasticNetCV(l1_ratio=0.1, selection='random', cv=cv, random_state=self.seed, normalize=False)
            model = ElasticNetCV(l1_ratio=0.1)
        else:
            model = LogisticRegressionCV(penalty='l1', cv=cv)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(basis, y)
            #coefs = model.feature_importances_
            _, coefs, _ = model.path(basis, y, l1_ration=model.l1_ratio_, alphas=model.alphas_)
        return model, coefs, model.mse_path_
            #y_pred_xgb = model.predict(basis)
            #mse = mean_squared_error(y,y_pred_xgb)
        #return model, coefs, mse

    def _get_selected_features(self):
        selected_features = []
        for _ in range(self.num_additions):
            feature = self._tournament_selection(self.current_features_)
            selected_features.append(feature)
        return selected_features

    def _get_coefficient_fitness(self, coefs, mse_path):
        mse = np.mean(mse_path, axis=1)
        #mse = mse_path
        r_squared = 1 - (mse / self.response_variance_)
        binary_coefs = coefs > self.fitness_threshold
        return binary_coefs.dot(r_squared)

    def _rank_by_coefficient(self, coefs, mse_path):
        fitness = self._get_coefficient_fitness(coefs, mse_path)
        for i, f in enumerate(self.current_features_):
            f.fitness = fitness[i]
        new_features = list(filter(lambda x: x.original_variable is True, self.current_features_))
        possible_features = list(filter(lambda x: x.original_variable is False, self.current_features_))
        possible_features.sort(key=lambda x: x.fitness, reverse=True)
        new_features.extend(possible_features[0:self.num_additions + 1])
        new_features.sort(key=lambda x: x.fitness, reverse=True)
        self.current_features_ = new_features

    def _remove_zeroed_features(self, model):
        remove_features = []
        for i, coef in enumerate(model.coef_):
            self.current_features_[i].fitness = math.fabs(coef)
            if self.current_features_[i].fitness <= self.fitness_threshold and not \
                    self.current_features_[i].original_variable:
                remove_features.append(self.current_features_[i])
        for f in remove_features:
            self.current_features_.remove(f)
        print('Removed ' + str(len(remove_features)) + ' features from population.')
        if self.verbose >= 2 and remove_features:
            print('Removed Features: ' + get_model_string(remove_features))

    def _update_fitness(self, y):
        basis, _ = self._get_current_basis()
        model, coefs, mse_path = self._build_linear_model(basis, y)
        if self.fitness_algorithm == ZERO_OUT:
            self._remove_zeroed_features(model)
        elif self.fitness_algorithm == COEFFICIENT_RANK:
            self._rank_by_coefficient(coefs, mse_path)

    def _uncorrelated(self, parents, new_feature):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            uncorr = True
            if type(parents) == list:
                for p in parents:
                    r, _ = pearsonr(new_feature.value, p.value)
                    if r > self.correlation_threshold:
                        uncorr = False
            else:
                r, _ = pearsonr(new_feature.value, parents.value)
                if r > self.correlation_threshold:
                    uncorr = False
        return uncorr

    def _tournament_selection(self, population):
        individuals = random.choices(population, k=2)
        individuals.sort(reverse=True, key=lambda x: x.fitness)
        if random.random() < self.tournament_probability:
            return individuals[0]
        else:
            return individuals[1]

    def _compose_features(self):
        new_feature_list = []
        for _ in range(self.num_additions):
            op = self.operators.get_random()[0]
            operator = self.operators.get(op)
            newFeatureAdded = False
            if operator.parity == 1:
                parent = self._tournament_selection(self.current_features_)
                new_feature_string = operator.string.format(parent.string)
                new_infix_string = operator.infix.format(parent.infix_string)
                new_feature_value = operator.operation(parent.value)            #the leaf node of the current tree
                new_feature = Feature(new_feature_value, new_feature_string, new_infix_string,
                                      size=parent.size + 1)
                new_feature_value_list = new_feature_value.tolist()
                new_feature_value_list = [ '%.5f' % elem for elem in new_feature_value_list ]
                if self._check_equal(new_feature_value_list):
                    if self.verbose >= 2:
                        print ("new_feature array is nearly constant for parity 01")
                    #print(parent)
                    #print(parent.value)
                    #print (new_feature.string)
                    #print (new_feature.value)
                elif self._uncorrelated(self.current_features_, new_feature):
                    new_feature_list.append(new_feature)
                    newFeatureAdded = True
            elif operator.parity == 2:
                parent1 = self._tournament_selection(self.current_features_)
                parent2 = self._tournament_selection(self.current_features_)
                new_feature_string = operator.string.format(parent1.string, parent2.string)
                new_infix_string = operator.infix.format(parent1.infix_string, parent2.infix_string)
                new_feature_value = operator.operation(parent1.value, parent2.value)
                new_feature = Feature(new_feature_value, new_feature_string, new_infix_string,
                                      size=parent1.size + parent2.size + 1)
                new_feature_value_list = new_feature_value.tolist()
                new_feature_value_list = [ '%.5f' % elem for elem in new_feature_value_list ]
                if self._check_equal(new_feature_value_list):   
                    if self.verbose >= 2:
                        print ("new_feature array is nearly constant for parity 02")
                elif self._uncorrelated(self.current_features_, new_feature):
                    if (parent1.infix_string != parent2.infix_string):
                        new_feature_list.append(new_feature)
                    #print("new feature :",new_feature)
                    #print("new feature type:",type(new_feature))
                        newFeatureAdded = True
                    #print ("new feature string",new_feature.string)
                    #print ("new feature value",new_feature.value)
                    #print(type(new_feature.value))
                    #print("new feature list",new_feature_value_list)
                    #print(type(new_feature_value_list))
                    #print ("parent 01",parent1.infix_string)
                    #print ("parent 01 type:",type(parent1.infix_string))
                    #print ("parent 02",parent2.infix_string)
                    #print ("parent 02 type:",type(parent2.infix_string))
            
            if (newFeatureAdded):

                if self.range_operators:
                    protected_range_operators = list(filter(lambda x: type(x) == RangeOperation and x.original_variable,
                                                            self.current_features_))
                    transitional_range_operators = list(filter(lambda x: type(x) == RangeOperation and not
                    x.original_variable,
                                                            self.current_features_))
                    if operator.infix_name == 'transition' and protected_range_operators:
                        parent = random.choice(protected_range_operators)
                        new_feature = deepcopy(parent)
                        new_feature.original_variable = False
                        new_feature_list.append(new_feature)
                    elif operator.infix_name == 'mutate' and transitional_range_operators:
                        parent = random.choice(transitional_range_operators)
                        new_feature = deepcopy(parent)
                        new_feature.mutate_parameters()
                        new_feature_list.append(new_feature)

        filtered_feature_list = list(filter(lambda x: x.size < 5, new_feature_list))
        

        ## chamika added the part to remove duplication
        filtered_feature_list_elements = []
        current_features_elements = []
        current_features_undup = []

        duplicate = []

        for ft in range(len(filtered_feature_list)):
            if filtered_feature_list[ft].string not in filtered_feature_list_elements:
                filtered_feature_list_elements.append(filtered_feature_list[ft].string)

        for ite in range (len(self.current_features_)):
            if self.current_features_[ite].string not in current_features_elements:
                current_features_elements.append(self.current_features_[ite].string)
                current_features_undup.append(self.current_features_[ite])
        
        duplicate = [item for item, count in collections.Counter(current_features_elements).items() if count > 1]

        if self.verbose >= 2:
            print ("Duplicated Features",duplicate)

        self.current_features_ = current_features_undup

        for found in range(len(filtered_feature_list_elements)):
            if filtered_feature_list_elements[found] not in current_features_elements:
                self.current_features_.append(filtered_feature_list[found])

        #self.current_features_ = list(set(self.current_features_))
        #self.current_features_.extend(filtered_feature_list)

        ### edited code ended

        if self.verbose >= 2:
            print('Adding ' + str(len(filtered_feature_list)) + ' features to population.')
            print('Added Features: ' + get_model_string(new_feature_list))

    def _score_model(self, y):
        if self.verbose >= 2:
            print('Scoring model with ' + str(len(self.current_features_)) + ' features.')
        basis, scaler = self._get_current_basis()
        model, coefs, _ = self._build_linear_model(basis, y)
        y_pred = model.predict(basis)
        score = mean_squared_error(y,y_pred)
        return score, model, scaler, y_pred

    def _swap_range_operators(self, X):
        for f in self.current_features_:
            if type(f) == RangeOperation and f.original_variable:
                self.current_features_.remove(f)
        for _ in range(self.range_operators):
            self.current_features_.append(RangeOperation(self.variable_type_indices, self.predictor_names_, X))

    def compute_operation(self, num_variables, stack, X):
        variables = []
        for _ in range(num_variables):
            variable_name = stack.pop()
            variable_index = self.predictor_names_.index(variable_name)
            variables.append(X[:, variable_index])
        operator = stack.pop()
        result = operator.operation(*variables)
        return result


class EFSRegressor(EvolutionaryFeatureSynthesis):

    def __init__(self, seed=None, fitness_algorithm=COEFFICIENT_RANK, method=None, max_gens=10,
                 num_additions=None, normalize=True, preserve_originals=True, tournament_probability=.9,
                 max_useless_steps=10, fitness_threshold=0.001, correlation_threshold=0.95, reinit_range_operators=3,
                 splits=3, time_series_cv=False, range_operators=0, variable_type_indices=None,
                 operators=default_operators, verbose=1):
        super().__init__(seed=seed, fitness_algorithm=fitness_algorithm, method=REGRESSION, max_gens=max_gens,
                         num_additions=num_additions, normalize=normalize, preserve_originals=preserve_originals,
                         tournament_probability=tournament_probability, max_useless_steps=max_useless_steps,
                         fitness_threshold=fitness_threshold, correlation_threshold=correlation_threshold,
                         reinit_range_operators=reinit_range_operators, splits=splits, time_series_cv=time_series_cv,
                         range_operators=range_operators, variable_type_indices=variable_type_indices,
                         operators=operators, verbose=verbose)


class EFSClassifier(EvolutionaryFeatureSynthesis):

    def __init__(self, seed=None, fitness_algorithm=COEFFICIENT_RANK, method=None, max_gens=10,
                 num_additions=None, normalize=True, preserve_originals=True, tournament_probability=.9,
                 max_useless_steps=10, fitness_threshold=.01, correlation_threshold=0.95, reinit_range_operators=3,
                 splits=3, time_series_cv=False, range_operators=0, variable_type_indices=None,
                 operators=default_operators, verbose=1):
        super().__init__(seed=seed, fitness_algorithm=fitness_algorithm, method=CLASSIFICATION, max_gens=max_gens,
                         num_additions=num_additions, normalize=normalize, preserve_originals=preserve_originals,
                         tournament_probability=tournament_probability, max_useless_steps=max_useless_steps,
                         fitness_threshold=fitness_threshold, correlation_threshold=correlation_threshold,
                         reinit_range_operators=reinit_range_operators, splits=splits, time_series_cv=time_series_cv,
                         range_operators=range_operators, variable_type_indices=variable_type_indices,
                         operators=operators, verbose=verbose)


