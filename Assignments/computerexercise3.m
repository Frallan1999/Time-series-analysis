%
% Time series analysis
% Computer exercise 3  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%% Prep 1

% The Kalman filter

% Simulate N samples of a process to test our code
N = 500;
A_sim = [1 -1.79 0.84];
e = randn(N,1);
y = filter(A_sim, 1, e); % y is our simulated data (tested via pzmap that we got the poles in the right places)

% Define the state space equations
A = eye(2);
Re = [0.1 0; 0 0]; % State covariance matrix (We suspect zero values for parameters that don't change over time)
Rw = 0; % Observation variance, we think 0 for simulated data

% Set some initial values
xt_t1 = [0 0]; % Initial state values: m0, expected value of x1 (WE CHANGED FROM xtt_1)
Rxx_1 = 10 * eye(2); % Initial state variance: Var X1, large V0 --> small trust in initial values

% Vectors to store values in
Xsave = zeros(2,N); % Stored states: For an AR(2) we have two hidden states, a1 and a2
ehat = zeros(1,N); % Prediction residual
yt1 = zeros(1,N) % One step prediction
yt2 = zeros(1,N); % Two step prediction

% The filter uses data up to time t-1 to predict value at t, then update
% using the prediction error. We start from t = 3, because we don't have
% the y values for t-3. We also stop at N - 2 as we don't have the true
% y-values for e.g. N + 1. 

for t=3:N-2
    Ct = [-y(t-1) -y(t-2)] % C_{t | t-1}
    yhat(t) = Ct * xt_t1; % y_t{t | t-1}
    ehat = y(t) - yhat(t); % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw; % R^{yy}_{t | t-1}
    Kt = % K_t
    xt_t =  %x_{t | t}
    Rxx = % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = % x_{t+1 | t}
    Rxx_1 = % R^{xx}_{t+1 | t}

end

%% Prep 2

%% 2.1.1 Recursive least squares

load tar2.dat %AR(2) with one time-dependent param and one constant

load thx.dat %The correct parameter trajectories

figure(1);
subplot(211);
plot(tar2)
subplot(212);
plot(thx);


%% 2.1.2 Recursive estimation of A(z)
clc; 
close all; 


X = recursiveAR(2); %Already implemented algorithm recursive AR
X.ForgettingFactor = 1; %Setting values for parts of the algo, lambda here
X.InitialA = [1 0 0]; %Initial values of A here

for kk=1:length(tar2)
    [Aest(kk,:), yhat(kk)] = step(X, tar2(kk));
end

%Testing different values for lamdba and plotting the result -- lambda 0.9
%seems to be the best, but very volatile estimate. 

%Question: Why do we get a lower value with lambda 1 than what we should
%expect? 
figure(1);
subplot(211);
plot(Aest(:,2:3))
subplot(212);
plot(thx);

%% 2.1.3 Finding optimal lambda via least squares
clc; 
close all;

n = 100;
lambda_line = linspace(0.85,1,n); %Creating a row vector of n different lambdas between 0.85 and 1
ls2 = zeros(n,1); %Initial ls estimator
yhat = zeros(n,1); %Storage for yhat

for i = 1:length(lambda_line) %Re-run the algorithm for all different lambda estimates
    reset(X);
    X.ForgettingFactor = lambda_line(i);
    X.InitialA = [1 0 0];
    for kk = 1:length(tar2)
        [~, yhat(kk)] = step(X,tar2(kk)); 
    end
    ls2(i) = sum((tar2-yhat).^2); %Saving the least square term for each lambda

end

plot(lambda_line,ls2) %The best lambda will have the lowest ls2 value
[minls2, minindex] = min(ls2);
minlambda = lambda_line(minindex)

%% 2.2 Kalman filtering of time series

clc; 
close all;



