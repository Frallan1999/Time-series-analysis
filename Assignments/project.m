%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)
%% 1. Introduction to the data
clear
close all
clc

load proj23.mat


%% 2.1: Studying the rain (org) data for El-Geneina
close all; 

% Redefining data 
rain_org = ElGeneina.rain_org;
rain_org_t = ElGeneina.rain_org_t;
rain = ElGeneina.rain;
rain_t = ElGeneina.rain_t;

% We start by plotting the rain_org data
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
% We now continue to model our rain as an AR(1) and reconstruct the rain
% using a Kalman filter 












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

