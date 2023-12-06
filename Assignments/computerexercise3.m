%
% Time series analysis
% Computer exercise 3  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%% Prep 1 - Kalman

% The Kalman filter

% Simulate N samples of a process to test our code
N = 1000;
A_sim = [1 -1.79 0.84];
e = 0.1*randn(N,1);
y = filter(A_sim, 1, e); % y is our simulated data (tested via pzmap that we got the poles in the right places)

% Define the state space equations
A = eye(2);
Re = [1e-6 0; 0 1e-6]; % State covariance matrix (We suspect zero values for parameters that don't change over time)
Rw = 0; % Observation variance, we think 0 for simulated data

% Set some initial values
xt_t1 = [0 0]'; % Initial state values: m0, expected value of x1 (WE CHANGED FROM xtt_1)
Rxx_1 = 10 * eye(2); % Initial state variance: Var X1, large V0 --> small trust in initial values

% Vectors to store values in
Xsave = zeros(2,N); % Stored states: For an AR(2) we have two hidden states, a1 and a2
ehat = zeros(1,N); % Prediction residual
yt1 = zeros(1,N); % One step prediction
yt2 = zeros(1,N); % Two step prediction

% The filter uses data up to time t-1 to predict value at t, then update
% using the prediction error. We start from t = 3, because we don't have
% the y values for t-3. We also stop at N - 2 as we don't have the true
% y-values for e.g. N + 1. 

for t=3:N-2
    Ct = [-y(t-1) -y(t-2)]; % C_{t | t-1}
    yhat(t) = Ct * xt_t1; % y_t{t | t-1}
    ehat(t) = y(t) - yhat(t); % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw; % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy; % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t); %x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1; % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t; % x_{t+1 | t}, don't forget to add B and U if needed
    Rxx_1 = A * Rxx * A' + Re; % R^{xx}_{t+1 | t}

    Xsave(:,t) = xt_t;

end

%% Prep 2 - Markov chain
clc;
close all;

% Simulate u_t where u_t is a Markov chain

% Define transition probabilities
p11 = 7/8;
p22 = 7/8;
p12 = 1/8;
p21 = 1/8; 

% Set number of interations
n = 5000;

% Preallocate a vector to store the states of the chain
u = zeros(1,n);

%Initialize state of the chain
state = 1;

% Simulate the chain
for t = 1:n
    u(t) = state; 

    % Draw a random number 
    r = rand();

    % Given state 1, transition to state 1 with prob p11
    % Given state 1, transition to state 2 with prob p12
    if state == 1 && r < p11
        state = 1; 
    elseif state == 1 && r >= p11
        state = -1;

    % Given state 2, transition to state 1 with prob p21
    % Given state 2, transition to state 2 with prob p22
    elseif state == -1 && r < p21
        state = 1;
    elseif state == -1 && r >= p21
        state = -1;

    end
end

% Plot state of chain at each iteration
plot(u)
xlabel('Iteration')
ylabel('State')

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

% The Kalman filter
N = length(tar2);
y = tar2;

% Define the state space equations
A = eye(2);
Re = [0.004 0; 0 0]; % State covariance matrix (zero values for parameters that don't change over time)
Rw = 1.25; % Observation variance, we think 0 for simulated data

% Set some initial values
xt_t1 = [0 0]'; % Initial state values: m0, expected value of x1 (WE CHANGED FROM xtt_1)
Rxx_1 = 10 * eye(2); % Initial state variance: Var X1, large V0 --> small trust in initial values

% Vectors to store values in
Xsave = zeros(2,N); % Stored states: For an AR(2) we have two hidden states, a1 and a2
ehat = zeros(1,N); % Prediction residual
yt1 = zeros(1,N); % One step prediction
yt2 = zeros(1,N); % Two step prediction

% The filter uses data up to time t-1 to predict value at t, then update
% using the prediction error. We start from t = 3, because we don't have
% the y values for t-3. We also stop at N - 2 as we don't have the true
% y-values for e.g. N + 1. 

for t=3:N %Note that we're not using N-2 here. Why? 
    Ct = [-y(t-1) -y(t-2)]; % C_{t | t-1}
    yhat(t) = Ct * xt_t1; % y_t{t | t-1}
    ehat(t) = y(t) - yhat(t); % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw; % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy; % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t); %x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1; % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t; % x_{t+1 | t}, don't forget to add B and U if needed
    Rxx_1 = A * Rxx * A' + Re; % R^{xx}_{t+1 | t}

    Xsave(:,t) = xt_t;

end

figure(1);
subplot(211);
plot(Xsave')
subplot(212);
plot(thx);

%% 2.3 Using Kalman filter for 2-step prediction
close all;

% Simulating data
rng(0);
N = 10000;
ee = 0.1*randn(N,1);
A0 = [1 -0.8 0.2];
y = filter(1, A0, ee);

% Define the state space equations.
A = eye(2);
Re = [1e-7 0; 0 1e-7];
Rw = 0.1; %We got a better predictions with Rw = 1 and 10. 

% Set some initial values
xt_t1 = [0 0]'; % Initial state values: m0, expected value of x1 (WE CHANGED FROM xtt_1)
Rxx_1 = 10 * eye(2); % Initial state variance: Var X1, large V0 --> small trust in initial values

% Vectors to store values in
Xsave = zeros(2,N); % Stored states: For an AR(2) we have two hidden states, a1 and a2
ehat = zeros(1,N); % Prediction residual
yt1 = zeros(1,N); % One step prediction
yt2 = zeros(1,N); % Two step prediction

% The filter uses data up to time t-1 to predict value at t, then update
% using the prediction error. We start from t = 3, because we don't have
% the y values for t-3. We also stop at N - 2 as we don't have the true
% y-values for e.g. N + 1. 

for t=3:N-2 %Why N-2 and not -1?
    Ct = [-y(t-1) -y(t-2)]; % C_{t | t-1}
    yhat(t) = Ct * xt_t1; % y_t{t | t-1}
    ehat(t) = y(t) - yhat(t); % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw; % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy; % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t); %x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1; % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t; % x_{t+1 | t}, don't forget to add B and U if needed
    Rxx_1 = A * Rxx * A' + Re; % R^{xx}_{t+1 | t}

    % Form 2-step prediction
    Ct1 = [-y(t) -y(t-1)]; % C_{t+1 | t}
    yt1(t+1) = Ct1 * xt_t; % y_{t+1 | t} = C_{t+1|t} x_{t|t}, %this holds as 
    % x_t is assumed to be constant over time, 
    % i.e. the parameters are not changing in this case

    Ct2 = [-yt1(t+1) -y(t)]; %C_{t+2 | t} %Why does this work - having yt1 here?
    yt2(t+2) = Ct2 * xt_t; % y_{t+2|t} = C_{t+2|t} x_{t|t}
    
    Xsave(:,t) = xt_t;

end

figure(1)
plot(y(end-100-2:end-2))
hold on
plot(yt1(end-100-1:end-1),'g')
plot(yt2(end-100:end),'r')
hold off
legend('y', 'k=1', 'k=2')

figure(2)
plot(Xsave')

%Compute sum of ehat for last 200 samples, why not all? 
sum(ehat(end-200:end))
ehat1 = y(2:end)' - yt1(1:end-1);
ehat2 = y(3:end)' - yt2(1:end-2);
ehat1_sumsq = sum(ehat1(end-200:end).^2)
ehat2_sumsq = sum(ehat2(end-200:end).^2)

%% 2.4 Quality control of a process - simulate the process

% Initial values
b = 20;
sigma2e = 1;
sigma2v = 4;
N = 5000;
rng(0);

% Simulate x as an AR(1)
e = sqrt(sigma2e)*randn(N,1);
A = [1 -1];
x = filter(1, A, e);

% Simulate y = x + bu + v
v = sqrt(sigma2v)*randn(N,1);
y = zeros(N,1);

for t = 1:N
    y(t) = x(t) + b*u(t) + v(t);
end

%% 2.4 Continued, find x and b

% Define the state space equations.
A = eye(2);
Re = [1e-2 0; 0 1e-2];
Rw = 1; 

% Set some initial values
xt_t1 = [0 15]'; % Initial state values
Rxx_1 = 10 * eye(2); % Initial state variance: Var X1, large V0 --> small trust in initial values

% Vectors to store values in
Xsave = zeros(2,N); % Stored states: For an AR(2) we have two hidden states, a1 and a2
ehat = zeros(1,N); % Prediction residual

for t=2:N
    Ct = [1 u(t)]; % C_{t | t-1}
    yhat(t) = Ct * xt_t1; % y_t{t | t-1} SHOULD WE INCORPORATE Vt?
    ehat(t) = y(t) - yhat(t); % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw; % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy; % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t); %x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1; % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t; % x_{t+1 | t}, don't forget to add B and U if needed
    Rxx_1 = A * Rxx * A' + Re; % R^{xx}_{t+1 | t}
    
    Xsave(:,t) = xt_t;

end

figure(1);
subplot(211);
plot(Xsave')
subplot(212);
plot(x);
yline(b);

%% 2.5.1 Recursive temperature modelling - Plot the data
close all;
clc;
clear;

load svedala94.mat

% Plot the data
figure(1)
y = svedala94;
T = linspace(datenum(1994,1,1),datenum(1994,12,31), length(svedala94)); % Get months on x-axis
plot(T,y);
datetick('x');
title('Raw data')

%% 2.5.1 Recursive temperature modelling - Differentiate

% Remove seasonality
D = [1 zeros(1,5) -1];
y_d = filter(D, 1, y); % Differentiate temperature with nabla 6
y_d = y_d(length(D):end); % Remove initial samples
T = linspace(datenum(1994,1,1),datenum(1994,12,31),length(y_d)); % Get months for x-axis

%Plot the data
figure(2)
plot(T,y_d)
datetick('x');
title('Differentiated data')

%% 2.5.2 Recursive temperature modelling - Fit ARMA


