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

%% 3. Transforming the data
% Transforming the NVDI (y) data as in B1

load proj23.mat

% Normalizing the data
y_org = ElGeneina.nvdi;
y_t = ElGeneina.nvdi_t;

max_data = 255;
min_data = 0;

y_all = 2*(y_org-min_data)/(max_data - min_data)-1;

% Split it 
ym = y_all(1:453,1);         % 70% for modelling
m_t = y_t(1:453,1);
ym_org = y_org(1:453,1);

yv = y_all(454:584,1);       % 20% for validation
v_t = y_t(454:584,1);
yv_org = y_org(454:584,1);

yt = y_all(585:end,1);        % 10% for test
t_t = y_t(585:end,1); 
yt_org = y_org(585:end,1); 

% Transformation of data 
y_log = log(y_all);
ym_log = log(ym);
yv_log = log(yv);
yt_log = log(yt);

% Useful for prediction
ym_yv = [ym; yv]; 
ym_yv_t = [m_t; v_t];
ym_yv_log = log(ym_yv);

modelLim = length(ym)+1; % Index for first data in validation set
testlim = length(ym_yv)+1; % Index for first data in test set

%% Naive model
% Let's say vegetation is the same as 1 year ago  
close all; 
clc;

noLags = 50; 
A = [1 zeros(1,35) 1];
C = [1];

model_naive = idpoly(A, [], C);
present(model_naive);

%% Test naive model on validation data (ver2)
close all; 
clc

k = 1;

yhat_k = filter(model_naive.a, model_naive.c, ym_yv_log); 
yhat_k = yhat_k(end-length(ym_yv)+1:end);

yhat_k_org = exp(yhat_k);
yhat_k_org = 1/2*(yhat_k_org+1)*(max_data - min_data)+min_data;
ym_yv_org = 1/2*(ym_yv+1)*(max_data - min_data)+min_data;

figure
plot(ym_yv_t, [ym_yv_org yhat_k_org] )
line( [ym_yv_t(modelLim) ym_yv_t(modelLim)], [0 200 ], 'Color','red','LineStyle',':' )
legend('NVDI', 'Naive model', 'Prediction starts')
title( sprintf('Predicted NVDI, validation data, y_{t+%i|t}', k) )
axis([ym_yv_t(length(ym)) ym_yv_t(end) min(yhat_k_org)*0.9 max(ym_yv_org)*1.1])

figure
hold on
plot(yhat_k_org)
plot(ym_yv_org)
legend('Naive model', 'Full NVDI data set')
hold off

%% 3.2.2 Model prediction
% Form the residual for the validation data

ehat = ym_yv_org - yhat_k_org;
ehat = ehat(modelLim:end);
var_ehat = var(ehat)
var_ehat_norm = var(ehat)/var(yv_org)


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
