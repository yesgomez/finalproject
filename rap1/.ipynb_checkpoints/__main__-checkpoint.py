from rap1.KfoldCrossValidation import *


# Declaring global stuff
dictionary = {'A':2, 'C':-2, 'T':3, 'G':-3, 'Y':1, 'N':0}
revdict = {2:'A', -2:'C', 3:'T', -3:'G', 1:'Y', 0:'N'}
declarations()
# K = int(sys.argv[1])
# positivefn = sys.argv[2]
# positive = []
# negativefn = sys.argv[3]
# negative = []
# i = 0
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)

# Split both datasets into K equal partitions (or "folds")
positive = cV.read(positivefn, 'Y')
negative = cV.read(negativefn, 'N')
pos = np.array(positive)
neg = np.array(negative)

combinedData = np.append(pos,neg)
l = len(pos)
fraction = np.round(int(l) / int(K))

pfolds = []
nfolds = []

# make list with lists of indices for a given fold of data as entries
for j in range(K):
	i = int(j*int(fraction))
	h = int(i+int(fraction))
	print(j,i,h)
	pfolds.append(pos[i:h])
	nfolds.append(neg[i:h])
if len(pfolds)==len(nfolds):
	print("Number of folds:",len(pfolds))

print ("\n--- Finding best classifier parameters using test data --- ")
# alpha = [0.0001, 0.001, 0.01, 1, 10, 100] # , 0.01, 1, 
# hls = list(range(2,17))
# best = []
# ix = np.round(0.8 * len(combinedData))
# fulltrain = []
# fullltest = []
# for a in alpha:
# for h in hls:
# 	train, test = cV.prep_data(0)
# 	predictions, score, ye = cV.NN_run(train, test, h)
# 	best.append(score)
# best = np.array(best) # This array lets you visually compare the scores from each param combination
# np.set_printoptions(precision=3)
# # best.resize((len(alpha),len(hls)))
# print(np.array(best), best.max())

print ("\n--- Running K fold cross validation using best settings ---")
netscores = []
neterrors= []

# Returns a trained model
Kfoldmodel, Kmlp = run_Kfold()

# Plotting ROC curve for different folds 
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
# plt.show()

# Use the average testing accuracy as the estimate of out-of-sample accuracy
avgScore = np.mean(netscores)
avgError= np.mean(neterrors)
print("\nThe average score for %s fold cross-validation is %s according to Scikit-learn and the accuracy is %s according to my own R^2 function." %(K, avgScore, avgError))

# Part 5. Testing on unlabelled data
ULdata = cV.read("rap1-lieb-test.txt",'')
ul = cV.translate(ULdata)
ul = pd.DataFrame(ul)
ULpredictions = Kmlp.predict(ul)
ULrawpredicts = Kmlp.predict_proba(ul)
rotate = ULrawpredicts.T[1]

with open("predictions_new.txt",'w') as w:
	w.write ("Sequence        \tProbability\n")
	for x, y in enumerate(ULdata):
		w.write("%s\t%s\n" %(y, rotate[x]))
print ("Done writing.")