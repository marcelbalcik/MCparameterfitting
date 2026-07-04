import json
import mcPolymer
import os
class modelInterpreter():
	def __init__(self, **kwargs):
		self.modelfile = kwargs.get('modelFile')
		self.add_recipe = False
		self.add_coefficient = False

	def addRecipe(self, **kwargs):
		self.recipe = kwargs.get('recipe',[])
		self.volume = kwargs.get('volume')
		with open('tmp_interpreter-add-recipe.txt', 'w') as file:
			for item in self.recipe:
				if 'mol' in item:
					speciesName = item['name']
					mol = float(item['mol'])
					lineStr = "[" + speciesName + "] = " + str(mol/self.volume) + '\n'
					file.write(lineStr)
		self.add_recipe = True

	def addCoefficient(self, **kwargs):
		name = kwargs.get('name')
		value = kwargs.get('value', None)
		linsStr= ""
		if value is not None:
			lineStr = name + " = " + str(value) + '\n'

		valueA = kwargs.get('A', None)
		valueEA = kwargs.get('EA', None)
		if (valueA is not None) and (valueEA is not None):
			lineStr = name + " = " + str(valueA) + "*exp(-" + str(valueEA) +"/RT)\n"

		if (not self.addCoefficient):
			with open('tmp_interpreter-add-coefficients.txt', 'w') as file:
				file.write(lineStr)
		else:
			with open('tmp_interpreter-add-coefficients.txt', 'a') as file:
				file.write(lineStr)
		self.add_coefficient = True

	def interpreteModelFile(self,**kwargs):
		if (self.add_coefficient):
			with open('tmp_interpreter-add-coefficients.txt', 'r') as f1:
				content_addCoefficient = f1.read()
		if (self.add_recipe):
			with open('tmp_interpreter-add-recipe.txt', 'r') as f2:
				content_addRecipe = f2.read()   
		with open(self.modelfile, 'r') as fm:
			content_model_file = fm.read()
		with open('tmp_merged_model_file.mcPolymer', 'w') as nf:
			if (self.add_recipe):
				nf.write(content_addRecipe)
			if (self.add_coefficient):
				nf.write(content_addCoefficient)
			nf.write(content_model_file)
		command = json.dumps({'mcPolymerCommand': 'interpreteModelFile', 'modelfile': 'tmp_merged_model_file.mcPolymer'})
		retStrInit = mcPolymer.py_interpreteModelFile(command)
		retStrInitClean = retStrInit.replace('\x00', '')
		jsonDict = json.loads(retStrInitClean)
		filename = self.modelfile + ".json"
		if os.path.exists(filename):
			os.remove(filename)
		os.rename('tmp_merged_model_file.mcPolymer.json', filename)
		if os.path.exists('tmp_interpreter-add-recipe.txt'):
			os.remove('tmp_interpreter-add-recipe.txt')
		if os.path.exists('tmp_interpreter-add-coefficients.txt'):
			os.remove('tmp_interpreter-add-coefficients.txt')
		if os.path.exists('tmp_merged_model_file.mcPolymer'):
			os.remove('tmp_merged_model_file.mcPolymer')
		return filename
