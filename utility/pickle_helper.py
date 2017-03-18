import pickle


class Pickle_Helper:

    def __init__(self, filePath):
        self.path = filePath
        self.objects = dict()

    def dump(self,obj,name):
        self.objects[name] = obj

    def save(self):
        f = open(self.path , "wb")
        pickle.dump(self.objects ,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        f = open(self.path , "rb")
        self.objects = pickle.load(f)
        f.close()
        return self.objects
    
    def get_path(self):
        return self.path
    
