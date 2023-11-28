import logging
from functools import wraps

def agent_says(text):
    print('-'*20)
    print(f"   Agent says: {text}")
    print('-'*20)
    logging.info(f"   Agent says: {text}")

def logthis(*args):
    print(*args)
    text = ' '.join([str(element) for element in args])
    logging.info(text)

def monitor_fn(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        logthis(f"Entering `{fn.__name__}`")
        res = fn(*args, **kwargs)
        logthis(f"Exiting  `{fn.__name__}`")
        return res
    return wrapper
