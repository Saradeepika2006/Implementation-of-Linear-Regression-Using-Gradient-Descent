# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data: Read CSV data, extract features (X) and target (y), and convert them to float.
2. Normalize features: Apply StandardScaler to scale both features and target for better training performance.
3. Define model: Implement gradient descent-based linear regression with bias term added to input features.
4. Train model: Iterate to minimize error and update model parameters (theta) using gradient descent.
5. Predict new value: Scale the new input data, apply the trained model, and get the scaled prediction.
6. Inverse scale result: Convert the scaled prediction back to the original scale and print it.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MOPURI SARADEEPIKA
RegisterNumber: 212224040201 
*/

        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
            X=np.c_[np.ones(len(X1)),X1]
            theta=np.zeros(X.shape[1]).reshape(-1,1)
            for _ in range(num_iters):
                predictions=(X).dot(theta).reshape(-1,1)
                errors=(predictions-y).reshape(-1,1)
                theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
                pass
            return theta
        
        
        data=pd.read_csv('/content/50_Startups.csv',header=None)
        print(data.head())
        
        
        X=(data.iloc[1:, :-2].values)
        print(X)
        
        
        X1=X.astype(float)
        scaler=StandardScaler()
        y=(data.iloc[1:,-1].values).reshape(-1,1)
        print(y)
        
        
        X1_Scaled=scaler.fit_transform(X1)
        Y1_Scaled=scaler.fit_transform(y)
        
        
        print(X1_Scaled)
        print(Y1_Scaled)
        
        theta=linear_regression(X1_Scaled,Y1_Scaled)
        new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
        new_Scaled=scaler.fit_transform(new_data)
        prediction=np.dot(np.append(1,new_Scaled),theta)
        prediction=prediction.reshape(-1,1)
        pre=scaler.inverse_transform(prediction)
        print(f"Predicted value: {pre}")
```
## Output:
<img width="562" height="120" alt="ML 8" src="https://github.com/user-attachments/assets/1b1ef8c7-fc9f-4738-bd0b-3d29bf69e930" />
<img width="1185" height="490" alt="ML 9" src="https://github.com/user-attachments/assets/d26ab715-cdd3-4ee9-ad67-322b76488353" />
<img width="320" height="390" alt="ML 10" src="https://github.com/user-attachments/assets/3bca5e60-d882-43a0-b829-94eddd3ac45c" />
<img width="861" height="483" alt="ML 11" src="https://github.com/user-attachments/assets/91fde567-d02d-4118-a794-77e1d7169766" />
<img width="892" height="407" alt="ML 12" src="https://github.com/user-attachments/assets/1d116893-ac6a-4328-882b-674c38e9f4b8" />
<img width="871" height="483" alt="ML 13" src="https://github.com/user-attachments/assets/2c6694d5-3e92-4faf-a7c1-b77115e7da5a" />
<img width="1151" height="489" alt="ML 14" src="https://github.com/user-attachments/assets/f9f74278-7bb4-41da-a8d6-8a0ea743f9e5" />
<img width="615" height="56" alt="ML 15" src="https://github.com/user-attachments/assets/1bbd52a3-36fb-468e-8e3d-1b004ae0bd7b" />
<img width="492" height="67" alt="ML 16" src="https://github.com/user-attachments/assets/87f7ac23-393c-43b8-8996-09e42ece3dd3" />











## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
