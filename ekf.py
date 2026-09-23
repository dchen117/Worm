import numpy as np
import cv2
import os
import csv
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import re

def compute_jacobian_F(x, dt):
    px, py, theta, v, omega = x.flatten()
    theta_rad = np.radians(theta)
    
    # Start with an identity matrix (ones on the diagonal)
    F = np.eye(5)
    
    # Fill in the upper triangular non-zero partial derivatives
    F[0, 2] = -v * dt * np.sin(theta_rad* (np.pi / 180.0))* (np.pi / 180.0)
    F[0, 3] = dt * np.cos(theta_rad)
    F[1, 2] = v * dt * np.cos(theta_rad* (np.pi / 180.0))* (np.pi / 180.0)
    F[1, 3] = dt * np.sin(theta_rad)
    F[2, 4] = dt
    
    return F

def predict(x, P, dt, Q):
    # Extract state variables
    px, py, theta, v, omega = x.flatten()
    
    # 1. State Transition (f)
    x_pred = np.array([
        px + v * np.cos(np.radians(theta)) * dt,
        py + v * np.sin(np.radians(theta)) * dt,
        theta + omega * dt,
        v,
        omega
    ]).reshape(5, 1)
    
    # 2. Compute Jacobian of f with respect to x (F)
    # (We will need to calculate these partial derivatives next)
    F = compute_jacobian_F(x, dt)
    
    # 3. Propagate Uncertainty
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

def update(x_pred, P_pred, z, R, H):
    # 1. Calculate Residual
    z_pred = H @ x_pred
    y = z - z_pred
    
    # Wrap the angular residual (index 2) to [-90, 90]
    y[2, 0] = ((y[2, 0] + 90) % 180) - 90

    S = H @ P_pred @ H.T + R

    mahalanobis_sq = (y.T @ np.linalg.inv(S) @ y)[0,0]

    if mahalanobis_sq > 9.21:  # Chi-square threshold for 3 DOF at 99% confidence
        print(f"Warning: Mahalanobis distance squared = {mahalanobis_sq:.2f} exceeds threshold. Measurement may be an outlier.")
        # bypass the update step and return the predicted state and covariance
        # return x_pred, P_pred

    # 2. Kalman Gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # 3. Update State and Covariance
    x_upd = x_pred + (K @ y)
    
    # Ensure final state angle is strictly [0, 180)
    x_upd[2, 0] %= 180
    
    I = np.eye(len(x_pred))
    P_upd = (I - K @ H) @ P_pred
    
    return x_upd, P_upd

def predict_endpoint(data, start_frame, x_init, P_init, Q, R, H):
    dt = 1.0 / 30.0
    x = x_init.copy()
    P = P_init.copy()
    
    # 1. Filtering Phase: process reality up to 'start_frame' (time t)
    for frame in range(start_frame + 1):
        z = data.loc[frame, ["x_cm", "y_cm", "theta_deg"]].to_numpy().reshape(3, 1)
        
        x_pred, P_pred = predict(x, P, dt, Q)
        x, P = update(x_pred, P_pred, z, R, H)
        
    # 2. Prediction Phase: blind simulation for 30 frames (t + 1 second)
    x_future = x.copy()
    P_future = P.copy()
    
    for _ in range(30):
        # The prediction output becomes the input for the next loop iteration
        x_future, P_future = predict(x_future, P_future, dt, Q)
        
    return x_future

