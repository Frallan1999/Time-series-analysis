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
% Examine outliers in modeling set -> looks fine
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

max_data = 255;
min_data = 0;

m = 2*(model-min_data)/(max_data - min_data)-1;
v = 2*(valid-min_data)/(max_data - min_data)-1;
t = 2*(test-min_data)/(max_data - min_data)-1;

plot(m_t,m)
%%  3.2 Model B1 - NVDI prediction without external input
% Transformation of data 
close all; 
clc; 

checkIfNormal(m,'modelling data')
subplot(121)
lambda_B1 = bcNormPlot(m)
title('Box-Cox normality plot for modelling data')
fprintf(['The Box-Cox curve is maximized at %4.2f.\n'], lambda_B1)
subplot(122)
normplot(m)

% Lets try with the log of the data (option 1)
m_log = log(m);
checkIfNormal(m_log,'modelling data');
figure();
plot(m_log);

%Lets try one over the square root of the data (option 2)
% m_sqrt= sqrt(m);
% checkIfNormal(m_sqrt,'modelling data');
% figure();
% plot(m_sqrt);

% Much better -> Lets continue with m_log 
v_log = log(v);
t_log = log(t);
%%  3.2 Model B1 - NVDI prediction without external input
% Differentiation of the data; 
clc; 
close all; 
noLags = 50;                % max up to N/4

plotACFnPACF(m_log,noLags, 'model data');

% Differentiate on season 36 with nabla (1-z^-36)
A36 = [1 zeros(1,35) -1];                       % Sets the season
m_diff = filter(A36,1,m_log);                   % Filter on seasonality 36 
m_diff = m_diff(length(A36):end);               % Omit initial samples
data = iddata(m_diff);                          % Create object for estimation
figure()
plot(m_diff);
plotACFnPACF(m_diff, noLags, "model data after differentiation with nabla36");  

% Differentiate on season 36 with (1-0.35*z^-36)
A36 = [1 zeros(1,35) -0.35];                     % Sets the season
m_diff = filter(A36,1,m_log);                    % Filter on seasonality 36 
m_diff = m_diff(length(A36):end);                % Omit initial samples
data = iddata(m_diff);                           % Create object for estimation
plotACFnPACF(m_diff, noLags, "model data with a36 = 0.35");  
figure()
plot(m_diff);

% Remove mean of differentiated data
m_diff_mean = m_diff - mean(m_diff);            % Note! This mean will prob not be same for validation and so on 
% m_diff_mean = m_diff;
figure()
plot(m_diff_mean)        

%%  3.2 Model B1 - NVDI prediction without external input
% Model the data after differentiating the data
close all; 
clc;
noLags = 50; 

% initial model - try AR(1)
A = [1 0];
C = [1];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_ar1 = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_ar1, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_ar1);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(1)');
plotNTdist(res.y);
% Looks white, but residual not normal --> can't 100 procent trust result.
% the residual is however t-dstriuted, even wider confidence interval for
% testing whiteness, so it is OK! 

%%  3.2 Model B1 - NVDI prediction without external input
% Model data without differentiation (incorporating the season)
close all; 
clc; 

noLags = 50; 

% Remove mean of logarithmised data
plot(m_log)
% m_mean = m_log - mean(m_log); 
% m_mean = m_log;

% initial model - based on information from differentiated
% A = [1 zeros(1,36)];
A = [1 0];
A = conv(A,A36);
C = [1];
data_m_log = iddata(m_log);
model_init = idpoly(A, [], C);
model_init.Structure.a.Free = [0 1 zeros(1,34) 1];
model_init.Structure.c.Free = [0 1 zeros(1,34) 1 1 1];
model_ar36 = pem(data_m_log, model_init);
res = resid(model_ar36, data_m_log);
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_ar36);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(36)');
plotNTdist(res.y);

%%  3.2.2 Model prediction (B1) 
close all; 
clc; 

k = 6;                  % sets number of steps prediction

% Solve the Diophantine equation and create predictions
[Fk, Gk] = polydiv(model_ar36.c, model_ar36.a, k);
throw = max(length(Gk), length(model_ar36.c));
yhat_k = filter(Gk, model_ar36.c, v_log);
yhat_k = yhat_k(throw:end);

% Transform prediction into original domain
yhat_k_org = exp(yhat_k);

% shiftK = round(mean(grpdelay(Gk, 1))) %THIS????

error = v_log(throw:end) - yhat_k;
error_org = v(throw:end) - yhat_k_org;  
var(error)
var(error_org)

% New domain plot
hold on
plot(yhat_k);
plot(v_log(throw:end));
hold off
basicPlot(error,noLags,'new')

% Original domain plot
figure()
hold on
plot(yhat_k_org);
plot(v(throw:end));
hold off
basicPlot(error_org,noLags,'org')

%% Create naive model 
