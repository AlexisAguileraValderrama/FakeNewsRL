import pickle
import random

class LocalDataManager:

    def __init__(self, localdata_path = ''):
      self.dataChunk = []
      self.current_data = ''
      self.datalist = []
      self.current_new = 0
      with open(localdata_path, "rb") as fp:   # Unpickling
        self.dataChunk = pickle.load(fp)
      print(len(self.dataChunk))
      self.data_count = 0

    def ChargeNewFromFile(self):
        current_data_dict = self.dataChunk[self.current_new]
        self.current_new = (self.current_new + 1) % len(self.dataChunk)
        a = len(self.dataChunk)
        self.title = current_data_dict['title']
        self.label = current_data_dict['label']
        self.datalist = current_data_dict['data']
        self.data_count = 0
        self.current_data = self.datalist[self.data_count]
        return True
      
    
    def GoNextArticle(self):
        self.data_count = self.data_count + 1
        if len(self.datalist) == self.data_count:
            return 0

        self.current_data = self.datalist[self.data_count]

        return 1
        

    def GetLoadedData(self):
        return self.title, self.label, self.current_data
    