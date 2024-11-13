New CATH classification model can be found in train/
All four classifier are based on the protein backbone encoder: ESM-1F. I follow this guidance to get the structure representation. (https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding#encoder-output-as-structure-representation)

Processed dataset:
train_embeddings.pt: ESM-1F structure representation of the train set
test_embeddings.pt: ESM-1F structure representation of the test set
valid_embeddings.pt: ESM-1F structure representation of the validation set

If you need the original dataset, you can download from https://zenodo.org/records/13643020. To process this original dataset to the representation, you can check the dataset class in the CATH_classifier.py

Models:
CATH_classifier.py: naive classifier using 2-layers MLP, arguments: --loss_type {"cross_entropy", "focal", "class_balanced"}, --task {"c", "a", "t","h"}
CATH_classifier_hierarchical.py: hierarchical classifier to get four layers classification results. Follow the idea from https://github.com/chen-bioinfo/HiPHD. arguments: --loss_type {"cross_entropy", "focal", "class_balanced"}
CATH_classifier_scl.py: train the naive classifier with supervised contrastive learning loss. arguments: --task {"c", "a", "t","h"} reference: https://arxiv.org/abs/2004.11362
CATH_calssifier_hierarchical_scl_v1.py: combination of the hierarchical classifier and scl loss. Might be interesting to explore https://arxiv.org/pdf/2402.00232.

Checkpoint folders:
Some trained models. Just for reference

To do list:
1. Train all classifiers, and compare the evluation results with the Chroma's original classifier. (Michael Xu)
2. Solve the class imbalance issue. (Michael Xu)
3. SOlve the over-fitting issue of some models. (Michael Xu)
4. If we have time, consider the Generalizability. How to generate samples not within any training classes. (Michael Xu, Yuxuan Liu)
5. Test the plain classifier's guidance performance. (Yuxuan Liu)
6. Improve the classifier's guidance performancce without training on noisy structure follow the idea from this paper: https://openreview.net/pdf?id=9DXXMXnIGm

