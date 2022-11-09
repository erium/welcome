import os, sys

method_names = {"linear": "backdoor.linear_regression",
                "strat":"backdoor.propensity_score_stratification",
                "match":"backdoor.propensity_score_matching",
                "ipw":"backdoor.propensity_score_weighting",
                "iv":"iv.instrumental_variable",
                "regdist":"iv.regression_discontinuity"}

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def estimate_causal_effect(model, identified_estimand, estimate_method, estimate_methods):
    method_name = method_names[estimate_method]
    if estimate_method == 'ipw':
        estimate = model.estimate_effect(identified_estimand,
            method_name=method_name, method_params={"weighting_scheme":"ips_weight"})
    elif estimate_method == 'regdist':
        estimate = model.estimate_effect(identified_estimand,
            method_name=method_name,
            method_params={'rd_variable_name':'residual sugar',
                        'rd_threshold_value':0.5,
                        'rd_bandwidth': 0.1})
    else:
        estimate = model.estimate_effect(identified_estimand,
            method_name=method_name)
    if estimate.value:
        estimate_methods[estimate_method] = estimate
    print("Causal Estimate is " + str(estimate.value))

def refute_causal_estimate(model, identified_estimand, estimate_methods, refute_data, is_treatment_binary):
    for estimate_name in estimate_methods:
        estimate = estimate_methods[estimate_name]
        refute_methods = {'random_common_cause': None, 'placebo_treatment': None, 'data_subset': None, 'unobserved_common_cause': None}

        if estimate_name not in ['iv', 'regdist']:
            print("Refuting with random common cause")
            with HiddenPrints():
                res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
                refute_methods['random_common_cause'] = res_random

        if estimate_name not in ['linear', 'regdist']:
            print("Refuting with placebo treatment")
            with HiddenPrints():
                res_placebo=model.refute_estimate(identified_estimand, estimate,
                    method_name="placebo_treatment_refuter", placebo_type="permute")
                refute_methods['placebo_treatment'] = res_placebo

        print("Refuting with data subset")
        with HiddenPrints():
            res_subset=model.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
            refute_methods['data_subset'] = res_subset
        
        if is_treatment_binary:
            print("Refuting with unobserved common cause")
            with HiddenPrints():
                res_unobserved_auto = model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                                    confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear")
                refute_methods['unobserved_common_cause'] = res_unobserved_auto
        
        refute_data[estimate_name] = refute_methods

def show_refute_results(refute_data):
    for estimate_method in refute_data:
        tests_passed = 0
        tests_failed = 0
        print(estimate_method)
        for refute_method in refute_data[estimate_method]:
            print(refute_method)
            if refute_data[estimate_method][refute_method] == None:
                print("None \n")
                continue
            print(refute_data[estimate_method][refute_method])
            refutation_result = refute_data[estimate_method][refute_method].refutation_result
            if refutation_result != None:
                if refutation_result['is_statistically_significant']:
                    tests_failed += 1
                else:
                    tests_passed += 1
        print("Statistical tests passed: ", tests_passed)
        print("Statistical tests failed: ", tests_failed)
        print('______________________________')