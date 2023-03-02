# Breast cancer prediction
## Abstact
Radiologists consider fine-grained characteristics of mammograms as well as patient-specific information before making the final diagnosis. Recent literature suggests that a similar strategy works forComputer Aided Diagnosis (CAD) models; multi-task learning with radiological and patient features as auxiliary classification tasks improves the model performance in breast cancer detection. Unfortunately, the additional labels that these learning paradigms require, such as patient
age, breast density, and lesion type, are often unavailable due to privacy restrictions and annotation costs. In this paper, we introduce a contrastive learning framework comprising a Lesion Contrastive Loss (LCL) and a Normal Contrastive Loss (NCL), which jointly encourage models to learn subtle variations beyond class labels in a self-supervised manner.The proposed loss functions effectively utilize the multi-view property of mammograms to sample contrastive image pairs. Unlike previous multitask learning approaches, our method improves cancer detection performance without additional annotations. Experimental results further demonstrate that the proposed losses produce discriminative intra-class features and reduce false positive rates in challenging cases.




