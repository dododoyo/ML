# Machine Learning 
- Has fundamentally three steps 

> Data Preprocessing  
  - import data
  - clean data
  - split into training and test set

> Modelling 
  - build model 
  - train model 
  - make predictions
  
> Evaluation
  - calculate performance metrics 
  - make a verdict 

## Data-Set
- It is of two types  Training Set and Testing Set

- 
## Feature Scaling 

- always applied to columns 

Feature scaling should be performed after splitting the data into training and test sets to prevent data leakage and ensure that the model generalizes well to unseen data. Here are the key reasons:

1. **Preventing Data Leakage**: If you scale the entire dataset before splitting, information from the test set can influence the scaling parameters (e.g., mean and standard deviation for standardization). This can lead to data leakage, where the model indirectly gains information about the test set, resulting in overly optimistic performance estimates.

2. **Ensuring Generalization**: The goal of a machine learning model is to generalize well to new, unseen data. By scaling the training set independently, the model learns scaling parameters based only on the training data. These parameters are then applied to scale the test set, simulating how the model would handle new data in a real-world scenario.

3. **Consistency in Scaling**: When you scale the training set first and then apply the same scaling parameters to the test set, you ensure that both sets are scaled consistently. This consistency is crucial for the model to make accurate predictions.

4. **Avoiding Overfitting**: Scaling the entire dataset before splitting can lead to overfitting, where the model performs well on the training data but poorly on new data. By scaling after splitting, you help the model to better capture the underlying patterns in the training data without being influenced by the test data.

In summary, performing feature scaling after splitting the data helps maintain the integrity of the test set, ensures consistent scaling, and promotes better generalization of the model. 

> normalization  // values will be between 0 and 1
- recommended for  normally destributed data

X' =  $\frac{X - Xmin}{Xmax-Xmin}$

> standardisation  // values will be beteween -3 and +3 
- works always 

X' =  $\frac{X - median}{S.D}$

we don't perform feature scaling on the dummy variables  

Great question! The reason for using [`fit_transform`] on the test set lies in the difference between fitting and transforming data.

### [`fit_transform`] on Training Data
- **Fitting**: When you call [`fit_transform`] on the training data, the scaler (or any other transformer) computes the necessary statistics (e.g., mean and standard deviation for standard scaling) from the training data.
- **Transforming**: After computing these statistics, it immediately applies the transformation to the training data.

### [`transform`] on Test Data
- **Transforming Only**: When you call [`transform`] on the test data, it uses the statistics computed from the training data (during the `fit` step) to transform the test data. It does not recompute the statistics, ensuring that the test data is scaled in the same way as the training data.

### Why This Approach?
1. **Consistency**: By using the same statistics (mean, standard deviation, etc.) from the training data to transform the test data, you ensure that both datasets are scaled consistently. This is crucial for maintaining the integrity of the model's performance evaluation.
2. **Avoiding Data Leakage**: If you were to fit the scaler on the test data, you would be introducing information from the test set into the training process, which can lead to overfitting and an overly optimistic evaluation of the model's performance.

```python
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

This ensures that both [`X_train`]and [`X_test`] are scaled consistently without any data leakage.
