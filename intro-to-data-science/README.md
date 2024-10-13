# intro-to-data-science

Repository for the project in introduction to data science.

The project consists of 2 parts:
  - replicating the article: https://www.sciencedirect.com/science/article/pii/S1877050918308548
  - improving on the results from the article

The dataset used is the PIMA Indians dataset which can be found here: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Project notes

### Article replication
- histograms of features, scatter plots, box plots
- 3 classifiers: SVM, naive Bayes, decision tree
- hyperparameter optimization done using grid search
- evaluation done using 100 random train test splits instead of fixing the seed
- results on the test set:
 
| Model                |  Precision | Recall    | F1 score | Accuracy   | AUC      | Hyperparameters used                                 |
| -------------------- | ---------- | --------- | -------- | ---------- | -------- | ---------------------------------------------------- |
| SVM                  | 0.517      | **0.734** | 0.603    | **0.768**  | **0.757**| ```{'C': 0.71, 'gamma': 'scale', 'kernel': 'poly'}```|
| Naive Bayes          | **0.600**  | 0.654     | **0.623**| 0.750      | 0.725    | ```{'var_smoothing': 0.0001}```|
| Decision tree        | 0.520      |0.648      |0.566     | 0.728      | 0.707    | ```{'ccp_alpha': 0, 'criterion': 'entropy', 'max_depth': 6, 'splitter': 'random'}```|

                 
### Improving the article results
- outlier removal - LOF algorithm, keeping 95% of the original examples
- feature engineering - automated using the autofeat library

- results:

| Model                                                  | Train accuracy |Dev accuracy |Dev precision | Dev recall |	Dev F1 score | 
| ------------------------------------------------------ | -------------- |------------- |	-------------- | ----------- |	------------- | 
| standard scaler, KNN                                   | 0.779    	    |0.760      | 0.694 	     |0.562	     | 0.618	    |
| minmax scaler, KNN                                     | 1.000 	        |0.761      | 0.682 	     |0.587 	   | 0.627 	    |
| select K best, standard scaler, KNN                    | 0.801 	        |0.763      | **0.706** 	 |0.543 	   | 0.610 	    |
| select K best, minmax scaler, KNN                      | 1.000 	        |0.757      | 0.679 	     |0.582 	   | 0.624 	    |
| select K best, polynomial features, standard scaler, LR| 0.862 	        |0.732      |	0.665 	     |0.467 	   | 0.546       |
| select K best, polynomial features, minmax scaler, LR  | 0.958 	        |0.690      | 0.555 		   |0.558 	   | 0.553       | 	
| RFE, polynomial features, standard scaler, LR          | 0.812 	        |0.751      | 0.653 		   |0.579 	   | 0.611       |
| RFE, polynomial features, minmax scaler, LR            | 0.798          |**0.775**  |	0.700		     |0.603 	   | 0.645	    |
| selectkbest, LGBM                                      | 0.999 	        |0.733      |	0.624 		   |0.585 	   | 0.601 	    |
| minmax scaler, random forest                           | 0.987 	        |0.754      | 0.643 		   |0.648 	   | 0.642 	    |
| standard scaler, random forest                         | 0.996          |0.754      |	0.660 		   |0.602 	   | 0.626 	    |
| select K best, standard scaler, random forest          | 0.967 	        |0.753      | 0.637 		   |0.664 	   | 0.648 	    |
| select K best, minmax scaler, random forest            | 0.943 	        |0.754      | 0.637 		   |0.679 	   | 0.655       | 	
| RFE, standard scaler, random forest                    | 0.879          |0.757      |	0.631 		   |**0.739**  | **0.678**   | 	
| RFE, minmax scaler, random forest                      | 0.946          |0.759      | 0.645 		   |0.701 	   | 0.670       | 	
| standard scaler, L1 regularized neural network         | 0.841 	        |0.732      |	0.606 	     |0.661 	   | 0.631 	    |

