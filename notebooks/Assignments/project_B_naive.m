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

yt = y_all(585:end,1);       % 10% for test
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

%% Test naive model on validation data
close all; 
clc;

ym_yv_org = 1/2*(ym_yv+1)*(max_data - min_data) + min_data;
yhat_org = length(ym_yv_org);

for t=37:length(ym_yv_org)
    yhat_org(t) = ym_yv_org(t-36);
end

hold on
plot(ym_yv_org(modelLim:end))
plot(yhat_org(modelLim:end))
hold off

error = ym_yv_org(modelLim:end)' - yhat_org(modelLim:end);
fprintf('  The variance of the residuals for the naive model on validation data is %5.2f.\n', var(error)')
fprintf('  The normalized variance of the residuals for the naive model on validation data is %5.2f.\n', var(error)/var(yv_org))

%% Test naive model on validation data
close all; 
clc;

yhat_org = length(y_org);

for t=37:length(y_org)
    yhat_org(t) = y_org(t-36);
end

hold on
plot(y_org(testlim:end))
plot(yhat_org(testlim:end))
hold off

error = y_org(testlim:end)' - yhat_org(testlim:end);
fprintf('  The variance of the residuals for the naive model on validation data is %5.2f.\n', var(error)')
fprintf('  The normalized variance of the residuals for the naive model on validation data is %5.2f.\n', var(error)/var(yt_org))

