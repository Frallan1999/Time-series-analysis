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
%% 2.1: Studying the rain (org) data for El-Geneina
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
%% 2.1: Studying the rain (org) data for El-Geneina
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
%% 2.1: Studying the rain (org) data for El-Geneina
% We now want continue to model our rain as an AR(1) and reconstruct the rain
% using a Kalman filter.

% Now that we are done with transforming the data, lets define it as y for
% simplicity 
y = log_rain_org;

% Define the state space equations.
a1 = 2;
A = a1*eye(3);     
Re = [zeros(1,3); zeros(1,3); zeros(1,3)];      % try different values
Rw = 1;                                         % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(log_rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1]; % C_{t | t-1}
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
plot(rain);
yline(b);

%% DO??? We want the mean to be zero
log_rain_org_m  = log_rain_org - mean(log_rain_org);

% Plotting the log_rain_org data
nbrLags = 50;
figure(3)
plot(rain_org_t, log_rain_org_m)
checkIfNormal(log_rain_org_m, 'ElGeneina rain_org')

%%
close all; 


log_data = log(ElGeneina.rain_org);
new_data = zeros(length(log_data),1);
for t=1:length(new_data)
    if log_data(t) == -inf 
        new_data(t) = 0;
    else 
        new_data(t) = log_data(t);
    end
end

plot(new_data)
basicPlot(new_data, nbrLags, 'Log ElGeneina.rain_org')

