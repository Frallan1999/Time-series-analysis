%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)
%% 1. Handling the data
% Examine data stationarity
% Split into modelling, validation and test data

clear
close all
clc

load proj23.mat

% Look at the data - deemed stationary!
figure(1)
plot(ElGeneina.nvdi_t,ElGeneina.nvdi)

% Split the data
n = length(ElGeneina.nvdi);

model = ElGeneina.nvdi(1:453,1);         % 70% for modelling
m_t = ElGeneina.nvdi_t(1:453,1);

valid = ElGeneina.nvdi(454:584,1);      % 20% for validation
v_t = ElGeneina.nvdi_t(454:584,1);

test = ElGeneina.nvdi(584:end,1);      % 10% for test
t_t = ElGeneina.nvdi_t(584:end,1); 

% Plot it
figure(2)
subplot(311)
plot(m_t,model);
title('Modeling set')
subplot(312)
plot(v_t,valid);
title('Validation set')
subplot(313)
plot(t_t,test);
title('Test set')

%% 1. Handling the data 
% Examine outliers in modeling set - looks fine
close all
clc

% Plot the ACF and TACF to see if tail-values have an impact on
% distribution - is this wrong interpretation?
noLags = 50;
figure(2)
acf(model, noLags, 0.02 ,1);
hold on
tacf(model, noLags, 0.02, 0.02 ,1);
hold off
title('ACF and TACF with alpha=0.02')

%% 1. Handling the data
% Normalize the data based on model set parameters
close all
clc

max = max(model);
min = min(model);

m = 2*(model-min)/(max - min)-1;
v = 2*(valid-min)/(max - min)-1;
t = 2*(test-min)/(max - min)-1;

plot(m_t,m)

%% Create naive model 


%% 2. NVDI prediction without external input

%% 3. NVDI prediction with external input