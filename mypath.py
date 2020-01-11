class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'ctchest':
            return '/'	# path of the dataset
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
