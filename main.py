import utilities
import plots
import featureAnalysis
import gaussianModels
import logisticRegression
import quadLogisticRegression
import svm
import svmPoly



if __name__=="__main__":
    #Loading the training dataset
    DTR,LTR=utilities.load_dataset("Test.txt")
    
   
    #Starting plots
    #plots.plot_hist(DTR, LTR)
    #plots.plot_scatter(DTR, LTR)
    #PCA
    #DTR_PCA, _=featureAnalysis.PCA(DTR, 2)
    #plots.plot_scatter_PCA(DTR_PCA, LTR)
    #LDA
    #DTR_LDA=featureAnalysis.LDA(DTR, LTR, 2)
    #plots.plot_hist_LDA(DTR_LDA, LTR)    
    
    #GAUSSIAN CLASSIFIERS
    
    #minDcfwithLMVG=gaussianModels.kFold(DTR, LTR)
    
    #LINEAR REGRESSION
    #logisticRegression.kFoldLR(DTR, LTR)
    #quadLogisticRegression.kFoldQLR(DTR, LTR)
    
    #SVM
    #svm.kFoldSvm(DTR, LTR)
    
    #SVMPoly
    svmPoly.kFoldSvmP(DTR, LTR)
