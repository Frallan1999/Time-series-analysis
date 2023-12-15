%
% Time series analysis
% Assignment 
%
%

clear; 
close all;
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)
%% 3. Modeling and validation for El-Geneina
clear
close all
clc

load proj23.mat

%% Naive model
% Lets say vegetation is the same as 1 year ago  
close all; 
clc; 

noLags = 50; 
A = [1 zeros(1,35) 1];
C = [1];

model_naive = idpoly(A, [], C);
present(model_naive);

%% Test naive model on validation data (ver2)
close all; 
yhat_k = filter(model_naive.a, model_naive.c, v); % Borde vi inte skicka in e här? 
yhat_k = yhat_k(length(model_naive.a):end)
error_org = v(length(model_naive.a):end) - yhat_k;
var(error_org)   % 0.0043

figure()
hold on
plot(yhat_k,'g');
plot(v(length(model_naive.a):end));
hold off
basicPlot(error_org,noLags,'Original domain not shifted')

hold on
plot(yhat_naive)
plot(ElGeneina.nvdi)
legend('Naive model', 'Full NVDI data set')
hold off

%% Test naive model on test data (ver2)
close all; 
yhat_k = filter(model_naive.a, model_naive.c, t);
yhat_k = yhat_k(length(model_naive.a):end)
error_org = t(length(model_naive.a):end) - yhat_k;
var(error_org)   % 0.0076

figure()
hold on
plot(yhat_k,'g');
plot(t(length(model_naive.a):end));
hold off
% basicPlot(error_org,noLags,'Original domain not shifted')


%% Test naive model on validation data
close all; 
clc; 
k = 37;                  % sets number of steps prediction
% very bad with k less than 37!!!

% Solve the Diophantine equation and create predictions
[Fk, Gk] = polydiv(model_naive.c, model_naive.a, k);
throw = max(length(Gk), length(model_naive.c));
yhat_k = filter(Gk, model_naive.c, v);
yhat_k = yhat_k(throw:end);

% It can be seen that the shift is IN GENERAL this (and is fun to then
% incorporate to be able to plot for both shifted and non shifted
if k == 1 || k == 2
    shift = k; 
else 
    shift = 3; 
end

% Create the errors (shifted and unshifted, original domain vs not) 
error_org_shifted = v(throw:end-shift) - yhat_k(1+shift:end);
error_org = v(throw:end) - yhat_k;
var(error_org)
var(error_org_shifted)

% Original domain plot (not shifted)
figure()
hold on
plot(yhat_k,'g');
plot(v(throw:end));
hold off
basicPlot(error_org,noLags,'Original domain not shifted')

% Original domain plot (shifted)
figure()
hold on
plot(yhat_k(1+shift:end),'g');
plot(v(throw:end-shift));
hold off
basicPlot(error_org,noLags,'Original domain')