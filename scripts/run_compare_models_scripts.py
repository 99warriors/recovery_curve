import plot_model_performances
import plot_full_model_posterior_parameters
import plot_predicted_patient_curves

if __name__ == '__main__':
    iter_module = sys.argv[1]
    the_iter = importlib.import_module(iter_module).the_iter
    plot_model_performances.plot_model_performances(the_iter)
    plot_full_model_posterior_parameters.plot_full_model_posterior_parameters(the_iter)
    plot_predicted_patient_curves.plot_predicted_patient_curves(the_iter)
