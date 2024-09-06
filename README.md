# IL-HTE_Educational_Item_Prediction_MIT_URTC_2024

[MIT_URTC_2024 - A Comparative Evaluation of the Predictive Performance of Artificial Neural Networks and Item Response Models for Educational Item-Level Heterogeneous Treatment Effects (3) (1).pdf](https://github.com/user-attachments/files/16913520/MIT_URTC_2024.-.A.Comparative.Evaluation.of.the.Predictive.Performance.of.Artificial.Neural.Networks.and.Item.Response.Models.for.Educational.Item-Level.Heterogeneous.Treatment.Effects.pdf)


The success of educational interventions is generally assessed through numerical approaches accounting for only total student scores. This approach, however, fails to consider item-specific effects, which can offer a more fine-grained perspective on the impact of the intervention. This work comparatively evaluates the ability of artificial neural network (ANN) approaches to recognize and evaluate heterogeneous treatment effects (HTE), which can allow for nuanced assistance that can better cater to individual student needs. Our work conducts a comparative analysis of artificial neural networks, specifically deep neural networks (DNN) and convolutional neural networks (CNN), to a traditional baseline explanatory item response model (EIRM). This study concludes that a neural network-based approach obtains higher accuracy and proves to be as competitive as traditional methods on most measured metrics. Our analysis of neural networks and traditional item-level modeling indicates that student baseline scores are most predictive of student responses, indicating that educational interventions should place more emphasis on assisting students in boosting their baseline abilities prior to an intervention. These results indicate that robust, complex modeling techniques such as neural networks can have utility in prediction in the space of educational interventions. 


# Code Structure
This repo contains scripts for training machine learning models to predict Item-Level Heterogeneous Treatment Effects (IL-HTE) using Logistic Regression, DNN, and CNN.

Each script:
1. Trains and saves a model to models/.
2. Calculates performance metrics (Accuracy, F1, MSE, AUC ROC).
3. Generates feature importance charts (SHAP, Absolute Coefficient Value).
