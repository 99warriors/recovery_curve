import plot_model_performances
import plot_full_model_posterior_parameters
import plot_predicted_patient_curves
import sys
import importlib

if __name__ == '__main__':
    iter_module_name = sys.argv[1]
    iter_module = importlib.import_module(iter_module_name)
    plot_model_performances.plot_model_performances(iter_module)
    plot_full_model_posterior_parameters.plot_full_model_posterior_parameters(iter_module)
    plot_predicted_patient_curves.plot_predicted_patient_curves(iter_module)
