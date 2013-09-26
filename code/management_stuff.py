"""
functions involved in using convert_to_print_form_functions, get_file_location_functions, and of course functions returning the associated objects, to call the function and save it 
"""

import os
import pdb
import functools

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

class call_and_save(object):
    """
    decorator for functions that does file based memoizing
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, inst, *args, **kwargs):
        location = inst.location_f(*args, **kwargs)
        key = inst.key_f(*args, **kwargs)
        full_path = '%s/%s' % (location, key)
        print len(full_path), full_path, 
        if os.path.exists(full_path) and not inst.to_recalculate:
            x = set_location_dec(set_hard_coded_key_dec(inst.read_f, key), location)(full_path)
        else:
            x = set_location_dec(set_hard_coded_key_dec(self.f, key), location)(inst, *args, **kwargs)
            if not os.path.exists(location):
                os.makedirs(location)
            inst.print_handler_f(x, full_path)
        return x

    def __get__(self, inst, cls):
        return functools.partial(self, inst)
        pdb.set_trace()
        self.inst = inst
        self.cls = cls
        return self.__call__

class call_and_key(object):
    """
    decorator that just sets the key 
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, inst, *args, **kwargs):
        key = inst.key_f(*args, **kwargs)
        return set_hard_coded_key_dec(self.f, key)(inst, *args, **kwargs)

    def __get__(self, inst, cls):
        #self.inst = inst
        #self.cls = cls
        #return self.__call__
        return functools.partial(self, inst)

class call_and_cache(object):
    """
    decorator for functions that does dictionary based memoizing
    """
    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, inst, *args, **kwargs):
        """
        call_and_cache __call__
        """
        key = inst.key_f(*args, **kwargs)
        try:
            x = self.cache[key]
            print 'FOUND'
        except KeyError:
            x = set_hard_coded_key_dec(self.f, key)(inst, *args, **kwargs)
            #x = self.f(inst, *args, **kwargs)
            self.cache[key] = x
            return x

    def __get__(self, inst, cls):

        return functools.partial(self, inst)

class save_object(object):
    """
    for object whose get_key and get_location are determined, aka a keyed_object, saves it
    
    """
    def __init__(self, get_location_f):
        pass

    def __call__(self, x):
        import os
        full_path = x.get_full_path()
        if not os.path.exists(full_path) or x.recalculate:
            x.print_handler_f(x, full_path)
        



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
        print type(self)
        raise NotImplementedError

    def get_key(self):
        try:
            return self.get_hard_coded_key()
        except AttributeError:
            return self.get_introspection_key()

    def get_location(self):
        return self.location

    def set_location(self, location):
        self.location = location

    def get_full_path(self):
        return '%s/%s' % (self.get_location(), self.get_key())

    def __cmp__(self, other):
        return self.get_key() == other.get_key()

def not_implemented_f(*args, **kwargs):
    raise NotImplementedError

class possibly_cached(keyed_object):

    def get_introspection_key(self):
        print self
        raise NotImplementedError

    def key_f(self, *args, **kwargs):
        raise NotImplementedError

    def location_f(self, *args, **kwargs):
        raise NotImplementedError

    print_handler_f = staticmethod(not_implemented_f)

    read_f = staticmethod(not_implemented_f)

    to_recalculate = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def full_path_f(self, *args, **kwargs):
        """
        refers to possible full path of the object returned by the function, not the function itself
        """
        return '%s/%s' % (self.location_f(*args, **kwargs), self.key_f(*args, **kwargs))


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



