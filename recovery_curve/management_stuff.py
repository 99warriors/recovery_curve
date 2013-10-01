"""
functions involved in using convert_to_print_form_functions, get_file_location_functions, and of course functions returning the associated objects, to call the function and save it 
"""

import os
import pdb
import functools
import numpy as np

class print_adapter(object):
    """
    class of adapters for convert_to_print_form_functions so that they all have same usage: 1. set location 2. call __call__ to write object to disk
    """
    def __init__(self, convert_to_print_form_f):
        self.convert_to_print_form_f = convert_to_print_form_f

    def set_location(self, location):
        self.location = location

    def get_location(self):
        return self.location

class string_adapter(print_adapter):

    def __call__(self, x, full_path):
        f = open(full_path, 'w')
        s = self.convert_to_print_form_f(x)
        assert type(s) == str
        f.write(s)
        f.close()

class StringIO_adapter(print_adapter):

    def __call__(self, x, full_path):
        f = open(full_path, 'w')
        import StringIO
        s = self.convert_to_print_form_f(x)
        assert type(s) == StringIO.StringIO
        f.write(s.getvalue())
        f.close()

class folder_adapter(print_adapter):
    """
    for the case when convert_to_print_form_f's output is a folder
    """
    def __call__(self, x, full_path):
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        self.convert_to_print_form_f(x, full_path)

class do_nothing_adapter(print_adapter):
    """
    for case when don't have to save anything because function did it already
    """

    def __init__(self):
        pass

    def __call__(self, x, full_path):
        pass

def compose(f,g):
    """
    function composition for functions taking only 1 argument
    """
    def composed(x):
        return f(g(x))

    return composed


class method_decorator(object):

    def __init__(self, f):
        self.f = f

    def __get__(self, inst, cls):
        return functools.partial(self, inst)

    def get_full_file_path(self, inst, *args, **kwargs):
        location = inst.location_f(*args, **kwargs)
        key = inst.key_f(*args, **kwargs)
        return '%s/%s' % (location, key)

    def get_full_pickle_path(self, inst, *args, **kwargs):
        location = inst.location_f(*args, **kwargs)
        key = inst.key_f(*args, **kwargs)
        return '%s/%s.pickle' % (location, key)

class read_from_file(method_decorator):

    def __call__(self, inst, *args, **kwargs):
        full_path = self.get_full_file_path(inst, *args, **kwargs)
        if os.path.exists(full_path):
            return inst.read_f(full_path)
        else:
            return self.f(inst, *args, **kwargs)

class save_to_file(method_decorator):

    def __call__(self, inst, *args, **kwargs):
        import os
        full_path = self.get_full_file_path(inst, *args, **kwargs)
        location = os.path.dirname(full_path)
        x = self.f(inst, *args, **kwargs)
        if not os.path.exists(location):
            os.makedirs(location)
        inst.print_handler_f(x, full_path)
        return x

class key(method_decorator):
    """
    adds information related to the key and full_file_path
    """
    def __call__(self, inst, *args, **kwargs):
        x = self.f(inst, *args, **kwargs)
        try:
            location = inst.location_f(*args, **kwargs)
        except NotImplementedError:
            # didn't define location_f for that factory
            pass
        else:
            try:
                x.set_location(location)
            except AttributeError:
                # returned object isn't a keyed_object
                pass
        key = inst.key_f(*args, **kwargs)
        try:
            x.set_hard_coded_key(key)
        except AttributeError:
            # returned object isn't a keyed_object
            pass
        try:
            x.set_creator(inst)
        except AttributeError:
            pass
        return x
            

class cache(method_decorator):

    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, inst, *args, **kwargs):
        key = inst.key_f(*args, **kwargs)
        try:
            x = self.cache[key]
        except KeyError:
            x = self.f(inst, *args, **kwargs)
            self.cache[key] = x
        return x

class read_from_pickle(method_decorator):

    def __call__(self, inst, *args, **kwargs):
        import pickle
        full_pickle_path = self.get_full_pickle_path(inst, *args, **kwargs)
        if os.path.exists(full_pickle_path):
            f = open(full_pickle_path, 'rb')
            x = pickle.load(f)
            f.close()
            return x
        else:
            return self.f(inst, *args, **kwargs)

class save_to_pickle(method_decorator):

    def __call__(self, inst, *args, **kwargs):
        import pickle
        x = self.f(inst, *args, **kwargs)
        full_pickle_path = self.get_full_pickle_path(inst, *args, **kwargs)
        location = os.path.dirname(full_pickle_path)
        if not os.path.exists(location):
            os.makedirs(location)
        f = open(full_pickle_path, 'wb')
        pickle.dump(x, f)
        f.close()
        return x

save_and_memoize = compose(key, compose(cache, save_to_file))
memoize = compose(key, cache)

def add_braces(f):
    def deced(*args, **kwargs):
        return '(%s)' % f(*args, **kwargs)
    return deced

class raise_if_na(object):
    """
    class method decorator that raises exception if return value isnan
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, inst, *args, **kwargs):
        ans = self.f(inst, *args, **kwargs)
        if np.isnan(ans):
            raise Exception
        return ans

    def __get__(self, inst, cls):
        return functools.partial(self, inst)


class keyed_object(object):
    """
    if object is created by caching factory, it will have key hard coded, and also location
    """

    def get_hard_coded_key(self):
        return self.hard_coded_key

    def set_hard_coded_key(self, hard_coded_key):
        print hard_coded_key
        assert type(hard_coded_key) == str
        self.hard_coded_key = hard_coded_key

    def get_introspection_key(self):
        raise NotImplementedError


    def get_key(self):
        try:
            return self.get_hard_coded_key()
        except AttributeError:
            #return self.get_introspection_key()
            return '(%s)' % self.get_introspection_key()

    def get_location(self):
        return self.location

    def set_location(self, location):
        self.location = location

    def set_creator(self, fp):
        self.fp = fp

    def get_creator(self):
        return self.fp

    def get_full_path(self):
        return '%s/%s' % (self.get_location(), self.get_key())

    def __cmp__(self, other):
        return self.get_key() == other.get_key()



def not_implemented_f(*args, **kwargs):
    raise NotImplementedError

class possibly_cached(keyed_object):

    def get_introspection_key(self):
        raise NotImplementedError

    def key_f(self, *args, **kwargs):
        raise NotImplementedError

    def location_f(self, *args, **kwargs):
        #print self
        raise NotImplementedError

    print_handler_f = staticmethod(not_implemented_f)

    read_f = staticmethod(not_implemented_f)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def full_path_f(self, *args, **kwargs):
        """
        refers to possible full path of the object returned by the function, not the function itself
        """
        return '%s/%s' % (self.location_f(*args, **kwargs), self.key_f(*args, **kwargs))

    def save(self, x):
        """
        assuming x has key and location set.  saves to file using the write_f for this factory
        for use when you only have access to the object returned by factory, not the arguments used to create it
        """
        self.print_handler_f(x, x.get_full_path())


class composed_getter(object):

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __get__(self, inst, cls):
        return getattr(self.f, self.attr_name)

class composed_factory(possibly_cached):

    def __init__(self, f, g):
        self.f, self.g = f,g

    def __call__(self, *args, **kwargs):
        return self.f(self.g(*args, **kwargs))

    def get_introspection_key(self):
        return '%s_%s' % (self.f.get_key(), self.g.get_key())

    def key_f(self, *args, **kwargs):
        return '%s_%s' % (self.get_key(), self.g.key_f(*args, **kwargs))

    print_handler_f = composed_getter('print_handler_f')

    read_f = composed_getter('read_f')



class save_factory_base(possibly_cached):
    """
    factory whose sole purpose is so that i can decorate __call__ with cache_and_save and save it
    pass in the pre-created item
    """
    def key_f(self, item):
        return item.get_key()

    @save_to_file
    @key
    def __call__(self, item):
        return item
        

def set_hard_coded_key_dec(f, hard_coded_key):

    def decorated_f(*args, **kwargs):
        x = f(*args, **kwargs)
        try:
            x.set_hard_coded_key(hard_coded_key)
        except AttributeError:
            pass
        return x

    return decorated_f

def set_location_dec(f, location):

    def decorated_f(*args, **kwargs):
        x = f(*args, **kwargs)
        try:
            x.set_location(location)
        except AttributeError, e:
            print e
            pass
        return x

    return decorated_f

def save_at_specified_path_dec(f, full_path):
    """
    decorator for factory f's __call__ that after __call__, saves object at specified path
    """
    def decorated_f(*args, **kwargs):
        x = f(*args, **kwargs)
        f.print_handler_f(x, full_path)
        return x

    return decorated_f

def None_if_exception(f, val):
    """
    decorator that returns val if an exception is raised by function call
    """
    def decorated_f(*args, **kwargs):
        try:
            x = f(*args, **kwargs)
        except Exception:
            return None
        else:
            return x
    return decorated_f


