from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np

# create some sample data
X = np.random.rand(100, 2)
y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# fit a RANSAC linear regression model
ransac = RANSACRegressor()
ransac.fit(X, y)

# fit a linear regression model using the inliers from the RANSAC model
inlier_mask = ransac.inlier_mask_
X_inliers = X[inlier_mask, :]
y_inliers = y[inlier_mask]
lr = LinearRegression()
lr.fit(X_inliers, y_inliers)

# get the estimated coefficients
params = np.concatenate(([lr.intercept_], lr.coef_))

# get the residuals
y_pred = lr.predict(X_inliers)
resid = y_inliers - y_pred

# get the covariance matrix of the estimated coefficients
cov_params = np.linalg.inv(X_inliers.T @ X_inliers) * np.sum(resid ** 2) / (X_inliers.shape[0] - X_inliers.shape[1])

print("Estimated coefficients:", params)
print("Covariance matrix of coefficients:", cov_params)

import statsmodels.api as sm
results = sm.OLS(X_inliers, y_inliers).fit()
print("Estimated coefficients:", results.params)
print("Covariance matrix of coefficients:", results.cov_params())