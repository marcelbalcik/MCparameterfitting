import mcPolymer
import json
import os
class kineticModel():
	def __init__(self, **kwargs):
		mcPolymer.py_startRandomGenerator()
		volume = kwargs.get('volume', None)
		print(volume)
		if volume is None:
			volume = 1.0
		command = json.dumps({'mcPolymerCommand': 'initKineticModel', 'volume': volume, 'numMolecules': kwargs.get('numMolecules'), 'temperature': kwargs.get('temperature'), 'modelfile': kwargs.get('modelFile')})
		retStrInit = mcPolymer.py_initKineticModel(command)
		retStrInitClean = retStrInit.replace('\x00', '')
		jsonDict = json.loads(retStrInitClean)
		self.controller_id = int(jsonDict['simulationID'])
		dirname = "mcPolymerSimulationresults-ID_" + str(self.controller_id)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		
		command = json.dumps({'simulationID': str(self.controller_id)})
		jsonStr0= mcPolymer.py_getVolume(command)
		
		jsonStr = jsonStr0.replace('\x00', '')
		jsonDict = json.loads(jsonStr)
		self.volumeInit = kwargs.get('volume')
		print('initial volume = ', self.volumeInit, ' L')
		cList = mcPolymer.py_getConcentrations(command)
		self.concentrationList = cList.replace('\x00', '')
		molList = mcPolymer.py_getMol(command)
		self.numMoleculesList = molList.replace('\x00', '')
		
	def runTo(self, **kwargs):
		command = json.dumps({'mcPolymerCommand': 'runTo', 'simulationID': str(self.controller_id), 'time': kwargs.get('time')})
		#print(mcPolymer.py_runTo(command))
		mcPolymer.py_runTo(command)
		command = json.dumps({'simulationID': str(self.controller_id)})
		cList = mcPolymer.py_getConcentrations(command)
		self.concentrationList = cList.replace('\x00', '')
		molList = mcPolymer.py_getMol(command)
		self.numMoleculesList = molList.replace('\x00', '')
		
	
	def updateTemperature(self, **kwargs):
		command = json.dumps({'mcPolymerCommand': 'setTemperature', 'simulationID': str(self.controller_id), 'temperature': kwargs.get('temperature')})
		print(mcPolymer.py_updateTemperature(command))
		
	def updateSolidsContent(self, **kwargs):
		command = json.dumps({'mcPolymerCommand': 'updateSolidsContent', 'simulationID': str(self.controller_id), 'solidsContent': kwargs.get('solidsContent')})
		print(mcPolymer.py_updateSolidsContent(command))
	
	def updateVolume(self, **kwargs):
		volume = kwargs.get('volume')
		command = json.dumps({'mcPolymerCommand': 'updateVolume', 'simulationID': str(self.controller_id), 'volume': volume})
		print(mcPolymer.py_updateVolume(command))
	
	def getConcentration(self, **kwargs):
		command = json.dumps({'simulationID': str(self.controller_id)})
		cList = mcPolymer.py_getConcentrations(command)
		concentrationList = cList.replace('\x00', '')
		speciesName = kwargs.get('species')
		jsonDictSpeciesConcentrations = json.loads(concentrationList)
		if (speciesName in jsonDictSpeciesConcentrations):
			return jsonDictSpeciesConcentrations[speciesName]
		else:
			print('error - species name ', speciesName, ' not valid')
			return 0.0
		
	def getMol(self, **kwargs):
		command = json.dumps({'simulationID': str(self.controller_id)})
		cList = mcPolymer.py_getMol(command)
		concentrationList = cList.replace('\x00', '')
		speciesName = kwargs.get('species')
		jsonDictSpeciesConcentrations = json.loads(concentrationList)
		if (speciesName in jsonDictSpeciesConcentrations):
			return jsonDictSpeciesConcentrations[speciesName]
		else:
			print('error - species name ', speciesName, ' not valid')
			return 0.0
			
	def feedMol(self, **kwargs):
		speciesName = kwargs.get('species')
		jsonDict = json.loads(self.numMoleculesList)
		if (speciesName in jsonDict):
			mol = kwargs.get('mol')
			command = json.dumps({'mcPolymerCommand': 'feedMol', 'simulationID': str(self.controller_id), 'species': kwargs.get('species'), 'mol': mol})
			print(mcPolymer.py_feedMol(command))
		else:
			print('error - species name ', speciesName, ' not valid')

	def updateMol(self, **kwargs):
		speciesName = kwargs.get('species')
		jsonDict = json.loads(self.numMolecules)
		if (speciesName in jsonDict):
			mol = kwargs.get('mol')
			command = json.dumps({'mcPolymerCommand': 'updateMol', 'simulationID': str(self.controller_id), 'species': kwargs.get('species'), 'mol': mol})
			print(mcPolymer.py_updateMol(command))
		else:
			print('error - species name ', speciesName, ' not valid')

	def exportPolymersCLD(self):
		command = json.dumps({'mcPolymerCommand': 'exportPolymersCLD', 'simulationID': str(self.controller_id)})		
		print(mcPolymer.py_exportPolymersCLD(command))
	
	def exportReactionsCount(self):
		command = json.dumps({'mcPolymerCommand': 'exportReactionsCount', 'simulationID': str(self.controller_id)})		
		mcPolymer.py_exportReactionsCount(command)
	
	def exportMMD(self, **kwargs):
		path = "mcPolymerSimulationresults-ID_" + str(self.controller_id)
		filename = kwargs.get('filename', None)
		if filename is None:
			filename = "MMD-Polymers"
		numRasterPoints = kwargs.get('numRasterPoints', None)
		if numRasterPoints is None:
			numRasterPoints = 800
		SECbroadeningParameter = kwargs.get('SECbroadeningParameter', None)
		if SECbroadeningParameter is not None:			
			command = json.dumps({'mcPolymerCommand': 'exportMMD', 'simulationID': str(self.controller_id), 'path': path, 'filename': filename, 'numRasterPoints': numRasterPoints, 'SECbroadeningParameter': SECbroadeningParameter})
		else:
			command = json.dumps({'mcPolymerCommand': 'exportMMD', 'simulationID': str(self.controller_id), 'path': path, 'filename': filename, 'numRasterPoints': numRasterPoints})
		print(mcPolymer.py_exportMMD(command))

	def exportPolymersBinary(self,**kwargs):
		filename = kwargs.get('filename')
		command = json.dumps({'mcPolymerCommand': 'exportPolymersCLD', 'simulationID': str(self.controller_id), 'filename': filename})
		print(mcPolymer.py_exportPolymersBinary(command))
		
	def getPolymerSegments(self, **kwargs):
		speciesName = kwargs.get('species')
		command = json.dumps({'mcPolymerCommand': 'getPolymerSegments', 'simulationID': str(self.controller_id)})	
		molList = mcPolymer.py_getPolymerSegments(command)
		molListSolidsContent = molList.replace('\x00', '')
		jsonDict = json.loads(molListSolidsContent)
		if (speciesName in jsonDict):
			return jsonDict[speciesName]
		else:
			print('error - species name ', speciesName, ' not valid')
			return 0.0
