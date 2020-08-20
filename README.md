# GMKSVM
Granular multiple kernel learning for identifying RNA-binding protein residues via integrating sequence and structure information

RNA-binding proteins play an important role in the biological process. However, the traditional experiment technology to predict RNA-binding residues is time-consuming and expensive, so the development of an effective computational approach can provide a strategy to solve this issue. In recent years, most of the computational approaches are constructed on protein sequence information, but the protein structure has not been considered. In this paper, we use a novel computational model of RNA-binding residues prediction, using protein sequence and structure information. Our hybrid features are encoded by local sequence and structure feature extraction models. Our predictor is built by employing the Granular Multiple Kernel Support Vector Machine with Repetitive Under-sampling (GMKSVM-RU). In order to evaluate our method, we use five-fold cross-validation on the RBP129, our method achieves better experimental performance with MCC of 0.3367 and accuracy of 88.84. In order to further evaluate our model, an independent data set (RBP60) is employed, and our method achieves MCC of 0.3921 and accuracy of 87.52. Above results demonstrate that integrating sequence and structure information is beneficial to improve the prediction ability of RNA-binding residues.


![](https://github.com/guofei-tju/GMKSVM/blob/master/GSVM-RU.pdf)
