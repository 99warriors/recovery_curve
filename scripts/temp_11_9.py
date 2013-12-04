








class model_comparer_f(possibly_cached):
    """
    hard code the loss functions used
    """

    def __init__(self, trainers, cv_f, loss_f, percentiles, times):
        self.trainers, self.cv_f, self.loss_f, self.percentiles, self.times = trainers, cv_f, loss_f, percentiles, times

    @save_to_file
    @memoize
    def __call__(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title(self.loss_f.get_key())
        for trainer in self.trainers:
            scores_getter = cross_validated_scores_f(trainer, self.cv_f, self.times)
            scores = scores_getter(data)
            _performance_series_f = performance_series_f(self.loss_f, self.percentiles, data, self.times)
            perfs = _performance_series_f(scores)
            add_performances_to_ax(ax, perfs, trainer.display_color, trainer.display_name)
        ax.set_xlim(-1.0,50)
        ax.set_ylim(-0.5,1.1)
        ax.legend()
        fig.show()
        return fig

    def get_introspection_key(self):
        return '%s_%s_%s' % (self.loss_f.get_key(), self.cv_f.get_key(), self.trainers[0].get_key())
        return '%s_%s_%s_%s' % ('bs', self.trainers.get_key(), self.cv_f.get_key(), self.loss_f.get_key())

    def key_f(self, data):
        return '%s_%s' % (self.get_key(), data.get_key())

    def location_f(self, data):
        return '%s/%s' % (global_stuff.data_home,'betweenmodel_perfs')

    print_handler_f = staticmethod(figure_to_pdf)

    read_f = staticmethod(not_implemented_f)
