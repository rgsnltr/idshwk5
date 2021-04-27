from sklearn.ensemble import RandomForestClassifier
import numpy as np

trainlist = []
testlist = []

def calcEntropy(name):
    letters=[i for i in name if i.isalpha()]
    l,counts=np.unique(letters,return_counts=True)
    all=sum(counts)
    pro=list(map(lambda x:x/all,counts))
    entropy=sum(-n*np.log(n) for n in pro)
    return entropy

def data_feature(name):
    dig_number=sum(i.isdigit() for i in name)
    return [len(name),dig_number,calcEntropy(name)]


class Domain:
	def __init__(self,_name,_label):
		self.name = _name
		self.label = _label


	def returnData(self):
		return data_feature(self.name)

	def returnLabel(self):
		if self.label == "notdga":
			return 0
		else:
			return 1
		
def initData(filename,file_type):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			if file_type == 'train':
				tokens=line.split(",")
				name=tokens[0]
				label=tokens[1]
				trainlist.append(Domain(name,label))
			else:
				testlist.append(line)

def main():
	initData("train.txt","train")
	initData("test.txt","test")
	#print(trainlist)
	#print(testlist)
	trainFeature = []
	trainLabel = []

	for item in trainlist:
		trainFeature.append(item.returnData())
		trainLabel.append(item.returnLabel())
	#print(featureMatrix)
	#print("Begin Training")
	clf = RandomForestClassifier(random_state=0)
	clf.fit(trainFeature,trainLabel)
	#print("Begin Predicting")
	testFeature=[data_feature(i) for i in testlist]
	predictList=clf.predict(testFeature)
	with open("result.txt","w") as f:
		for i in range(len(predictList)):
			if predictList[i]==0:
				f.write(testlist[i]+",notdga\n")
			else:
				f.write(testlist[i]+",dga\n")

if __name__ == '__main__':
	main()
