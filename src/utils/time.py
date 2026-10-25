from functools import wraps
from time import time
from tdpy.verbosity import print

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()  # [s]
        result = f(*args, **kw)
        te = time()  # [s]
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap