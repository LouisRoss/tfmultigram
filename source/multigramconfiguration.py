import os
import json

path = '/record/'
basefoldername = 'configurations'

class MultigramConfiguration:
    def __init__(self, config_name, configuration=None):
        if configuration is None:
            self.config_name = config_name
            if not self.config_name.endswith(".json"):
                self.config_name += ".json"

            self.valid = False
            self.configuration = {'name':'<none>'}
            configfilename = path + basefoldername + '/' + self.config_name
            if os.path.exists(configfilename):
                with open(configfilename, 'r') as configfile:
                    self.configuration = json.load(configfile)
                    self.valid = True
        else:
            """ For unit testing.
            """
            self.configuration = configuration
            self.valid = True

        self.Initialize()


    def Initialize(self):
        self.name = None
        self.iteration_count = None
        self.layer_size = None
        self.thickness = None
        self.delaydepth = None
        self.threshold = None
        self.interconnectCount = None
        self.outputwidth = None
        self.selected_initializer = None

    def Save(self, datafolder):
        if os.path.exists(datafolder):
            with open(datafolder + '/configuration.json', 'w') as configfile:
                json.dump(self.configuration, configfile)

    def GetName(self) -> str:
        if not self.name:
            self.name = self.configuration['name']

        return self.name
    
    def SetName(self, name: str):
        self.name = name
    
    def GetIterationCount(self) -> int:
        if not self.iteration_count:
            if self.valid and 'iterationCount' in self.configuration:
                self.iteration_count = self.configuration['iterationCount']
            else:
                self.iteration_count = 0
        
        return self.iteration_count

    def SetIterationCount(self, iteration: int):
        self.iteration_count = iteration

    def GetLayerSize(self) -> int:
        if not self.layer_size:
            if self.valid and 'layerSize' in self.configuration:
                self.layer_size = self.configuration['layerSize']
            else:
                self.layer_size = 0
        
        return self.layer_size
    
    def SetLayerSize(self, layer_size: int):
        self.layer_size = layer_size

    def GetThickness(self) -> int:
        if not self.thickness:
            if self.valid and 'thickness' in self.configuration:
                self.thickness = self.configuration['thickness']
            else:
                self.thickness = 0
        
        return self.thickness
    
    def SetThickness(self, thickness: int):
        self.thickness = thickness

    def GetDelayDepth(self) -> int:
        if not self.delaydepth:
            if self.valid and 'delayDepth' in self.configuration:
                self.delaydepth = self.configuration['delayDepth']
            else:
                self.delaydepth = 1
        
        return self.delaydepth
    
    def SetDelayDepth(self, delaydepth: int):
        self.delaydepth = delaydepth

    def GetThreshold(self) -> int:
        if not self.threshold:
            if self.valid and 'threshold' in self.configuration:
                self.threshold = self.configuration['threshold']
            else:
                self.threshold = 0
        
        return self.threshold
    
    def SetThreshold(self, threshold: int):
        self.threshold = threshold

    def GetInterconnectCount(self) -> int:
        if not self.interconnectCount:
            if self.valid and 'interconnectCount' in self.configuration:
                self.interconnectCount = self.configuration['interconnectCount']
            else:
                self.interconnectCount = 0
        
        return self.interconnectCount
    
    def SetInterconnectCount(self, interconnectCount: int):
        self.interconnectCount = interconnectCount

    def GetOutputWidth(self) -> int:
        if not self.outputwidth:
            if self.valid and 'outputWidth' in self.configuration:
                self.outputwidth = self.configuration['outputWidth']
            else:
                self.outputwidth = 0
        
        return self.outputwidth
    
    def SetOutputWidth(self, outputwidth: int):
        self.outputwidth = outputwidth

    def GetSelectedInitializer(self) -> int:
        if not self.selected_initializer:
            if self.valid and 'selectedInitializer' in self.configuration:
                self.selected_initializer = self.configuration['selectedInitializer']
            else:
                self.selected_initializer = 0
        
        return self.selected_initializer
    
    def SetSelectedInitializer(self, initializer_number: int):
        self.selected_initializer = initializer_number

    def GetInitializers(self) -> list[str]:
        if self.valid and 'initializers' in self.configuration:
            return self.configuration['initializers']
        
        return []
    
    