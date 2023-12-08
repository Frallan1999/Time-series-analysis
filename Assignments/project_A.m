%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)
%% 1. Introduction to the data
clear
close all
clc

load proj23.mat
%% 2.1.1 Studying the rain data for El-Geneina
close all; 

% Saving data in new variables
rain_org = ElGeneina.rain_org;
rain_org_t = ElGeneina.rain_org_t;
rain = ElGeneina.rain;
rain_t = ElGeneina.rain_t;

%% 2.1.1: Gaussian analysis of original rain data
close all; 

nbrLags = 50;
figure(1)
plot(rain_org_t, rain_org)
title('rain org')
subplot(121)
lambda_max = bcNormPlot(rain_org,1)
title('Box Cox plot of rain org')

fprintf('The Box-Cox curve is maximized at %4.2f. This is very close to zero, and thus suggests that a log-transform might be helpful.\n', lambda_max)
subplot(122)
normplot(rain_org)
checkIfNormal(rain_org, 'ElGeneina rain org');

% Looking at the Normal probability plot. The rain_org data does not look gaussian at all.
% Looking at the BJ curve we see a maximization close to zeo -> suggesting
% a log transform might be helpful

%% 2.1.1: Gaussian analysis of original rain data
% Adding constant and log transforming the data
close all; 

% Adding constant to data 
constant = 1;
rain_org_c = rain_org + constant;

% Log transforming data with constant
log_rain_org = log(rain_org_c);

% Plotting the log_rain_org data
nbrLags = 50;
figure(1)
plot(rain_org_t, log_rain_org)
checkIfNormal(log_rain_org, 'ElGeneina rain_org')

% It is still not Gaussian, but we look away and say yey 
%% 2.1.1: Gaussian analysis of original rain data
% Removing the mean 
log_rain_org_m  = log_rain_org - mean(log_rain_org);

% Plotting the log_rain_org data
nbrLags = 50;
figure(3)
plot(rain_org_t, log_rain_org_m)
checkIfNormal(log_rain_org_m, 'ElGeneina rain_org') 
%% 2.1.2: Finding a reasonable initial a1
% LOG DATA
% We want to model our rain as an AR(1) and reconstruct the rain
% using a Kalman filter. To get an idea of what the a parameter in the
% AR(1) process could be, we start by trying to model our log_rain_org as an
% AR(1) to get an idea
close all; 

% We do a basic plot
basicPlot(log_rain_org, nbrLags, 'log rain org')
% See a lot of seasonality in ACF, disregard this and try to model as AR(1)

model_init = idpoly([1 0], [], []);
data = iddata(log_rain_org);
model_ar = pem(data, model_init);
present(model_ar)
res = myFilter(model_ar.c, model_ar.a, log_rain_org);
basicPlot(res, nbrLags, 'res');

%% 2.1.2: Finding a reasonable initial a1 
% ORIGINAL DATA
% We want to model our rain as an AR(1) and reconstruct the rain
% using a Kalman filter. To get an idea of what the a parameter in the
% AR(1) process could be, we start by trying to model our log_rain_org as an
% AR(1) to get an idea
close all; 

% We do a basic plot
basicPlot(rain_org, nbrLags, 'rain org')
% See a lot of seasonality in ACF, disregard this and try to model as AR(1)

model_init = idpoly([1 0], [], []);
data = iddata(rain_org);
model_ar = pem(data, model_init);
present(model_ar)
res = myFilter(model_ar.c, model_ar.a, rain_org);
basicPlot(res, nbrLags, 'res');

%% 2.1.3: Kalman reconstruction
% Now that we are done with transforming the data and have found an 
% inital estimate for a1, lets go ahead with a Kalman reconstruction. 
close all;
%y = log_rain_org;                               % Redefine the data as y for simplicity
%y = y_sim;         
y = rain_org

% Define the state space equations. Value of a1 optimized for ORIGINAL
% (non-log) data.
a1 = 0.2491;
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-4 0 0; 1e-6 0 0; 0 1e-6 0];            % try different values
Rw = 1e-4;                                       % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(log_rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1];                               % C_{t | t-1}
    yhat(t) = Ct * xt_t1;                       % y_t{t | t-1} 
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1} (reffered to as y_tilde in project)

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw;                % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy;                     % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);                  % x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1;              % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t;                           % x_{t+1 | t} this is our AR(1) process 
    Rxx_1 = A * Rxx * A' + Re;                  % R^{xx}_{t+1 | t}
       
    Xsave(:,t) = xt_t;
end

% We would like to store this in an vector as in the interpolated case 
rain_kalman = zeros(3*length(rain_org),1); 
for k = 1:length(Xsave)
    rain_kalman(3*k) = Xsave(1,k);
    rain_kalman(3*k-1) = Xsave(2,k);
    rain_kalman(3*k-2) = Xsave(3,k);
end

%No negative rain!
for k = 1:length(rain_kalman)
    if (rain_kalman(k) < 0)
        rain_kalman(k) = 0;
    end
end

%% 2.1.3: Kalman reconstruction
% ORIGINAL DATA
% Plotting the results
figure(1);
subplot(311);
plot(rain_t, rain_kalman)
subplot(312);
plot(rain_org_t, rain_org)
subplot(313);
plot(rain_t, rain)

sum(rain_kalman)                 
sum(rain_org)                               
abs(sum(rain_kalman)-sum(rain_org))

%% 2.1.3: Kalman reconstruction
% Plotting the results
% LOG DATA
figure(1);
subplot(311);
plot(rain_t, rain_kalman)
subplot(312);
plot(rain_org_t, log_rain_org)
subplot(313);
plot(rain_t, log(rain+constant))

sum(rain_kalman)                 
sum(log_rain_org)                               
abs(sum(rain_kalman)-sum(log_rain_org))
%% 2.1.4: Simulated data
% Generate the hidden states x_t+1 = a1 * x_t + et

N1 = 3*N;
extraN = 100;
A1 = [1 -a1]; 
e = randn(N1+extraN,1); 
x_sim = filter(1, A1, e); x_sim = x_sim(extraN+1:end);

for i = 1:N1
    if(x_sim(i)<0) 
        x_sim(i) = 0;
    end
end

y_sim = zeros(N,1);
v = randn(N,1);

for i = 1:N
    y_sim(i) = x_sim(3*i) + x_sim(3*i-1) + x_sim(3*i-2) + v(i);
end

%% 2.1.4: Simulated data
% Plot simulation vs reality

figure(1)
plot(rain_kalman)
figure(2)
plot(x_sim)

sum(rain_kalman)
sum(x_sim)
sum(v)
