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

- It is of two types Training Set and Testing Set

-

## Feature Scaling

- always applied to columns

Feature scaling should be performed after splitting the data into training and test sets to prevent data leakage and ensure that the model generalizes well to unseen data. Here are the key reasons:

1. **Preventing Data Leakage**: If you scale the entire dataset before splitting, information from the test set can influence the scaling parameters (e.g., mean and standard deviation for standardization). This can lead to data leakage, where the model indirectly gains information about the test set, resulting in overly optimistic performance estimates.

2. **Ensuring Generalization**: The goal of a machine learning model is to generalize well to new, unseen data. By scaling the training set independently, the model learns scaling parameters based only on the training data. These parameters are then applied to scale the test set, simulating how the model would handle new data in a real-world scenario.

3. **Consistency in Scaling**: When you scale the training set first and then apply the same scaling parameters to the test set, you ensure that both sets are scaled consistently. This consistency is crucial for the model to make accurate predictions.

4. **Avoiding Overfitting**: Scaling the entire dataset before splitting can lead to overfitting, where the model performs well on the training data but poorly on new data. By scaling after splitting, you help the model to better capture the underlying patterns in the training data without being influenced by the test data.

In summary, performing feature scaling after splitting the data helps maintain the integrity of the test set, ensures consistent scaling, and promotes better generalization of the model.

> normalization // values will be between 0 and 1

- recommended for normally destributed data

X' = $\frac{X - Xmin}{Xmax-Xmin}$

> standardisation // values will be beteween -3 and +3

- works always

X' = $\frac{X - median}{S.D}$

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

## Simple Linear Regression

### Ordinary Least Squares

if we have y = b0 + b1x
we want to have b0 and b1 such that sum[(yi -y)^2] is 
minimized this is called `Oridinary Least Squares`. 

### Statistical Significance

Imagine you have a big jar of colorful candies – red, blue, and green. You want to know if there are more red candies than the other colors.

You ask your friend to close their eyes and pick out a handful of candies. Let's say they get mostly red ones! Does that mean there are definitely more red candies in the jar? Not so fast! It could be, or it could just be a lucky handful.

Statistical significance is like a detective's magnifying glass for situations like this. It helps us figure out if what we see (lots of red candies!) is truly meaningful or just random chance.

We do some math with the candies your friend picked and compare it to all the candies in the jar. If the math says it's very unlikely to get that many red candies just by chance, then we can be more confident that there really are more red candies in the jar. That's "statistically significant"!

In machine learning, we use this detective work to see if patterns we find in data are real or just flukes. If something is statistically significant, we can be more confident that our machine learning model is learning something true and not just making lucky guesses.

### p-value

A p-value is a measure used in statistical hypothesis testing to determine the significance of the results. It represents the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true.

The name "p-value" stands for "probability value." It represents the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true. The "p" in p-value is derived from the word "probability."

### Key Points:

- **Null Hypothesis (H0)**: The default assumption that there is no effect or no difference.
- **Alternative Hypothesis (H1)**: The assumption that there is an effect or a difference.
- **Low p-value (< 0.05)**: Indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
- **High p-value (> 0.05)**: Indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.

`In the context of machine learning, a low p-value for a predictor suggests that the predictor is statistically significant and likely contributes meaningfully to the model. Conversely, a high p-value suggests that the predictor may not be significant and could be considered for removal from the model.`

In machine learning, the predictor with the highest p-value signifies that it is the least statistically significant variable in the model. A high p-value indicates that there is a high probability that the observed effect could have occurred by chance, suggesting that the variable may not be a meaningful predictor of the outcome.

### Filter Variables

- We can't use every variable in our model. somethings are just not that important
- Garbage in Garbage out
- Understand every variables is hard

### Fitting a Model

Fitting a model in machine learning is like teaching a robot to recognize patterns. Imagine you have a bunch of pictures of cats and dogs. You want the robot to learn how to tell the difference between them.

1. **Collect Data**: First, you show the robot many pictures of cats and dogs.
2. **Teach the Robot**: The robot looks at these pictures and tries to find patterns, like "cats usually have pointy ears" or "dogs often have floppy ears."
3. **Practice**: The robot keeps practicing with these pictures until it gets really good at telling cats from dogs.

When the robot has learned well, we say the model is "fitted" to the data. Now, when you show the robot a new picture, it can use what it learned to guess if it's a cat or a dog.

### Methods for Building Models(Filtering Variables)

> All in

- Include everything, You have to, all determine Equally,preparing for backward elimination. . .

> Backward Elimination
> `step-1` - select significance model (SL = 0.05)
> `step-2` - fit the full model with all possible predictors
> `step-3` - remove the predictor with the highest p-value
> `step-4` - fit the model without removed variable

> Forward Selection
> `step-1` - select significance model (SL = 0.05)
> `step-2` - fit every single variable with a simple regression model
> `step-3` - select one with the lowest p-value to be your model
> `step-4` - add all the remaining and create all possible 2-variable linear regression Models
> `step-5` - select one model with lowest p-value of all possible 2-variable linear regression Models

- and so on . . . . . until we see fit
  `step-6` - ones the lowest p-value becomes less than SL we stop this process

> Bidirection Elimination
> `step-1` - select significance model (SLENTER = 0.05, and SLSTAY = 0.05)
> `step-2` - perform forward elimination with SLENTER
> `step-3` - perform backward elimination with SLSTAY
> `step-4` - stop when no new variables can (enter and leave)

> All Possible Models
> `step-1` - select a criterion of goodness of fit (eg Akaike criteron)
> `step-2` - construct all possible Regression models 2\*\*N - 1 total combinations
> `step-3` - select one with the best criterion

The **All Possible Models** approach is a comprehensive method for selecting the best regression model by evaluating every possible combination of predictors. Here's a detailed explanation of each step:

#### Step-by-Step Explanation

1. **Select a Criterion of Goodness of Fit**:

   - Before constructing any models, you need to decide on a criterion to evaluate how well each model fits the data. Common criteria include:
     - **Akaike Information Criterion (AIC)**: Balances model fit and complexity, penalizing models with more predictors.
     - **Bayesian Information Criterion (BIC)**: Similar to AIC but with a stronger penalty for models with more predictors.
     - **Adjusted R-squared**: Adjusts the R-squared value based on the number of predictors, preventing overfitting.

2. **Construct All Possible Regression Models**:

   - For [`N`] predictors, there are `2^N - 1` possible combinations of these predictors (excluding the model with no predictors).
   - For example, if you have 3 predictors (A, B, and C), the possible models are:
     - Model 1: A
     - Model 2: B
     - Model 3: C
     - Model 4: A + B
     - Model 5: A + C
     - Model 6: B + C
     - Model 7: A + B + C
   - This step involves fitting a regression model for each of these combinations.

3. **Select the Best Model Based on the Criterion**:
   - After fitting all possible models, you evaluate each one using the chosen criterion of goodness of fit.
   - The model that has the best (e.g., lowest AIC or BIC) criterion value is selected as the final model.
   - This ensures that the selected model provides the best balance between fit and complexity according to the chosen criterion.

#### Advantages and Disadvantages

- **Advantages**:

  - Comprehensive: Evaluates all possible models, ensuring the best model is selected.
  - Objective: Uses a clear criterion for model selection.

- **Disadvantages**:
  - Computationally Intensive: For a large number of predictors, the number of possible models grows exponentially, making this method computationally expensive.
  - Overfitting Risk: Without proper criteria, there's a risk of selecting overly complex models.

This method is best suited for situations where the number of predictors is relatively small, allowing for a thorough evaluation of all possible models.

> You don't need to use this filtering methods libraries  implement them 

## Support Linear Regeression

