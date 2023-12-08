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
%% 2.1.1 Studying the rain (org) data for El-Geneina - Plotting the data
close all; 

% Saving data in new variables
rain_org = ElGeneina.rain_org;
rain_org_t = ElGeneina.rain_org_t;
rain = ElGeneina.rain;
rain_t = ElGeneina.rain_t;

% We start with normal analyzis of the rain_org data
nbrLags = 50;
figure(1)
plot(rain_org_t, rain_org)
title('rain org')
figure(2)
lambda_max = bcNormPlot(rain_org,1)
title('Box Cox plot of rain org')

fprintf('The Box-Cox curve is maximized at %4.2f. This is very close to zero, and thus suggests that a log-transform might be helpful.\n', lambda_max)
checkIfNormal(rain_org, 'ElGeneina rain org')

% Looking at the Normal probability plot. The rain_org data does not look gaussian at all.
% Looking at the BJ curve we see a maximization close to zeo -> suggesting
% a log transform might be helpful
%% 2.1.1 Studying the rain (org) data for El-Geneina - Log transform to make it more Gaussian
% Adding constant, log transforming the data, and removing the mean 
close all; 

% Adding constant to data 
constant = 1;
rain_org_c = rain_org + constant;

% Log transforming data with constant
log_rain_org = log(rain_org_c);

% Plotting the log_rain_org data
nbrLags = 50
figure(1)
plot(rain_org_t, log_rain_org)
checkIfNormal(log_rain_org, 'ElGeneina rain_org')

% It is still not Gaussian, but we look away and say yey 
%% 2.1.1 Studying the rain (org) data for El-Geneina - Zero-mean
log_rain_org_m  = log_rain_org - mean(log_rain_org);

% Plotting the log_rain_org data
nbrLags = 50;
figure(3)
plot(rain_org_t, log_rain_org_m)
checkIfNormal(log_rain_org_m, 'ElGeneina rain_org')

%% 2.1.1 Studying the rain (org) data for El-Geneina - Starting values for Kalman filter
% We now want continue to model our rain as an AR(1) and reconstruct the rain
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
 

%% 2.1.2 Kalman filter to find hidden states
% Now that we are done with transforming the data, lets define it as y for
% simplicity 
close all;

y = log_rain_org;

% Define the state space equations.
a1 = 0.9;
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-2 0 0; 0 1e-6  0; 0 0 1e-2];           % try different values
Rw = 0.5;                                         % try different values

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
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1}

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
rain_kalman = zeros(length(Xsave(1,:))*length(Xsave(:,1)),1);
k = 1; 
for T=1:length(Xsave(1,:))
    for t=1:length(Xsave(:,1))
        rain_kalman(k) = Xsave(t,T);
        k = k+1; 
    end

end

%% Plotting the results
figure(1);
subplot(311);
plot(rain_t, rain_kalman)
subplot(312);
plot(rain_org_t, log_rain_org)
subplot(313);
plot(rain_t, log(rain+constant))

sum(rain_kalman(rain_kalman>0))   % not removed mean
sum(log_rain_org)                 % not removed mean

%% Do we like negative rain? NO -> one option is to put to zero, other to move up? 
% here lets try putting it to zero :) 
close all;

rain_kalman_pos = zeros(length(rain_kalman),1);
for t=1:length(rain_kalman_pos)
    if rain_kalman(t) < 0
        rain_kalman_pos(t) = 0;
    else 
        rain_kalman_pos(t) = rain_kalman(t);
    end
end

figure(1);
subplot(311);
hold on
plot(rain_t, rain_kalman_pos)
scatter(rain_t, rain_kalman_pos)
hold off
subplot(312);
hold on
plot(rain_org_t, log_rain_org)
scatter(rain_org_t, log_rain_org)
hold off
subplot(313);
hold on
plot(rain_t, log(rain+constant))
scatter(rain_t, log_rain)
hold off

sum(rain_kalman_pos)
sum(log_rain_org)

