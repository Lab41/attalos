
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import json
import threading
from Queue import Queue

class JsonWorker(threading.Thread):
    def __init__(self, name, in_queue, out_queue, func=None):
        super(JsonWorker, self).__init__()
        
        self.name = name
        self.in_q = in_queue
        self.out_q = out_queue
        self.load_func = func or self.load_json
        
    @staticmethod
    def load_json(fp):
        with open(fp) as infile:
            return json.load(infile)
        
    def run(self):
        try:
            while not self.in_q.empty():
                fp = self.in_q.get()
                data = self.load_func(fp)
                self.out_q.put(data)
        except Exception as e:
            raise(e)


def load_combine_json_dir(dir_path, func, thread_count=4, verbose=True):
    """Function to read in a directory with json files. Each 
    json file will be loaded and formatted with func
    
    Parameters
    ----------
    dir_path : string
        path to json directory
    func : funciton
        function that accepts one argument to load json and
        returns a dictionary
    thread_count : int
        number of worker threads to use. Default is 8
    verbose : bool
        when True will print messages about data loading
    
    Returns
    --------
    dictionary
    """
    in_q = Queue()
    out_q = Queue()
    
    n_files = 0
    for d, subd, files in os.walk(dir_path):
        for f in files:
            if f.endswith('json'):
                in_q.put(os.path.join(d, f))
                n_files += 1
    threads = []
    n_threads =  thread_count if n_files > thread_count else n_files
    
    if n_threads == 0:
        if verbose is True:
            print("No json files found")
        return 
    for i in range(n_threads):
        worker = JsonWorker(str(i + 1), in_q, out_q, func=func)
        worker.start()
        threads.append(worker)
        
    for worker in threads:
        worker.join()
    
    out_json = {}
    while not out_q.empty():
        portion = out_q.get()
        out_json.update(portion)
    if verbose is True:
        print("Loaded {0} files".format(n_files))
    return out_json
    