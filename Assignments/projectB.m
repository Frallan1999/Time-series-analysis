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
%% 3.1 Dividing and cleaning of data set
% Examine the data's stationarity
% Split into modelling, validation and test data

% Look at the data - deemed stationary!
figure(1)
plot(ElGeneina.nvdi_t,ElGeneina.nvdi)

% Split the data
n = length(ElGeneina.nvdi);

model = ElGeneina.nvdi(1:453,1);         % 70% for modelling
m_t = ElGeneina.nvdi_t(1:453,1);

valid = ElGeneina.nvdi(454:584,1);       % 20% for validation
v_t = ElGeneina.nvdi_t(454:584,1);

test = ElGeneina.nvdi(584:end,1);        % 10% for test
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

%% 3.1 Dividing and cleaning of data set
% Examine outliers in modeling set - looks fine
close all
clc

% Plot the ACF and TACF to see if tail-values have an impact on
% distribution - is this wrong interpretation?
nbrLags = 100;
subplot(121)
acf(model, nbrLags, 0.02 ,1);
hold on
tacf(model, nbrLags, 0.02, 0.02 ,1);
hold off
title('ACF and TACF with alpha=0.02')

subplot(122)
acf(model, nbrLags, 0.01 ,1);
hold on
tacf(model, nbrLags, 0.01, 0.01 ,1);
hold off
title('ACF and TACF with alpha=0.01')

%%  3.1 Dividing and cleaning of data set
% Normalize the data based on model set parameters
close all
clc

max_m = max(model);
min_m = min(model);

m = 2*(model-min_m)/(max_m - min_m)-1;
v = 2*(valid-min_m)/(max_m - min_m)-1;
t = 2*(test-min_m)/(max_m - min_m)-1;

plot(m_t,m)


%% 2. NVDI prediction without external input
% Start by plotting the data
close all
clc

basicPlot(m,100,'Modeling data')
checkIfNormal(m,'Modeling set','D',0.05);

% Two reflections:
% 1. Strong season of 12 to be handled - differentiate
% 2. PACF suggests maybe AR(1)?

m_d = myFilter([1 zeros(1,35) -0.7],1,m);

subplot(121)
plot(m)
title('Original data')
subplot(122)
plot(m_d)
title('Differentiated data')

basicPlot(m_d,50,'Differentiated data')


%% 3. NVDI prediction with external input


%% Create naive model 
