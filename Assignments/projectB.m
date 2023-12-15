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

% WE NEED TO SEE IF LOG IS THE RIGHT WAY TO GO --> FULL was previously m!!!!!!
full = ElGeneina.nvdi;

checkIfNormal(full,'modelling data')
subplot(121)
lambda_B1 = bcNormPlot(full)
title('Box-Cox normality plot for modelling data')
fprintf(['The Box-Cox curve is maximized at %4.2f.\n'], lambda_B1)
subplot(122)
normplot(full)

% Lets try with the log of the data (option 1)
m_log = log(full);
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
mean(m_diff)

% Differentiate on season 36 with (1-0.35*z^-36)
A36 = [1 zeros(1,35) -0.35];                     % Sets the season
m_diff = filter(A36,1,m_log);                    % Filter on seasonality 36 
m_diff = m_diff(length(A36):end);                % Omit initial samples
data = iddata(m_diff);                           % Create object for estimation
plotACFnPACF(m_diff, noLags, "model data with a36 = 0.35");  
figure()
plot(m_diff);
mean(m_diff)

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

% initial model - try a1
A = [1 0];
C = [1];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_arma = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_arma, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(1)');
plotNTdist(res.y);
% FPE: 0.04411 and Monti: 19.11 < 36.42
% OBS! Residual not normal --> can't 100 procent trust result.
% the residual is however t-dstriuted, even wider confidence interval for
% testing whiteness, so it is OK! 
% ACF & PACF --> Maybe include c36 or/and c3? 

% Although white, a bit of a peak at 36 still -> try a1 & c36
A = [1 0];
C = [1 zeros(1,36)];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.c.Free = [0 zeros(1,35) 1];        % added ones C changed
model_arma = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_arma, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1 and c36 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1 and c36 modelling');
plotNTdist(res.y);
% FPE: 0.04292 and Monti: 26.56 < 36.42
% All significant

% Although white, a bit of a peak at 36 still -> try a1, c3 & c36
A = [1 0];
C = [1 zeros(1,36)];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.c.Free = [0 0 0 1 zeros(1,32) 1];        % added ones C changed
model_arma = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_arma, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1, c3, and c36 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1, c3, and c36 modelling');
plotNTdist(res.y);
% FPE: 0.04254 and Monti: 17.93 < 36.42
% All significant (Best so far) 

% Last was good, but could we simplify model by removing c36 without loosing
% much? 
A = [1 0];
C = [1 0 0 1];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.c.Free = [0 0 0 1];        % added ones C changed
model_arma = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_arma, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1 and c3 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1 and c3 modelling');
plotNTdist(res.y);
% FPE: 0.04409 and Monti: 12.52 < 36.42
% c36 seems unnecessary (probabky even more when incorporating this season
% in the model)

%%  3.2 Model B1 - NVDI prediction without external input
% Model data without differentiation (incorporating the season)
close all; 
clc; 

noLags = 50; 

% Remove mean of logarithmised data
% m_mean = m_log - mean(m_log); 
% m_mean = m_log;

% initial model - based on information from differentiated
A = [1 0];
A = conv([1 zeros(1,35) -1], A);
% C = [1]
C = [1 0 0 1];
% C = [1 zeros(1,36)];
 
data_m_log = iddata(m_log);

model_init = idpoly(A, [], C);
model_init.Structure.a.Free = [0 1 zeros(1,34) 1 0];        
model_init.Structure.c.Free = [zeros(1,3) 1];        
% model_init.Structure.c.Free = [0 0 0 1 zeros(1,32) 1];        
model_sarima = pem(data_m_log, model_init);
res = resid(model_sarima, data_m_log);
plotACFnPACF(res.y, noLags, "Residual for SARIMA modelling");
figure()
present(model_sarima);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals from SARIMA modelling');
plotNTdist(res.y);
% FPE: 0.04658 and Monti: 8.89 < 36.42
% A "jump" in the normplot...

%% 3.2.2. Model prediction
% Lets do a one step and save the variance
k = 1;      % prediction step 

[Fk, Gk] = polydiv(model_sarima.c, model_sarima.a, k );  
throw = max(length(Gk), length(model_sarima.c));
yhat_1 = filter(Gk, model_sarima.c, v_log);                             
yhat_1 = yhat_1(throw:end);   
yhat_1_org = exp(yhat_1);
var_1 = var(v(throw:end) - yhat_1_org)
var_test = var(v_log(throw:end) - yhat_1)

basicPlot(e, 50, 'e')

e = v(throw:end) - yhat_1_org;          % Calculating our error terms that we expect to behave like an MA
eps = myFilter(Fk, 1, e);               % Removing the MA-part, to see if white. 
basicPlot(eps, 50, 'Stationary prediction error') % It is!


%%  3.2.2 Model prediction (B1) 
close all; 
clc; 
k = 45;                  % sets number of steps prediction

% Solve the Diophantine equation and create predictions
[Fk, Gk] = polydiv(model_sarima.c, model_sarima.a, k);
throw = max(length(Gk), length(model_sarima.c));
yhat_k = filter(Gk, model_sarima.c, v_log);
yhat_k = yhat_k(throw:end);

% Transform prediction into original domain
yhat_k_org = exp(yhat_k);

% It can be seen that the shift is IN GENERAL this (and is fun to then
% incorporate to be able to plot for both shifted and non shifted
if k == 1 || k == 2
    shift = k; 
else 
    shift = 3; 
end

% Create the errors (shifted and unshifted, original domain vs not) 
error_shifted = v_log(throw:end-shift) - yhat_k(1+shift:end);
error_org_shifted = v(throw:end-shift) - yhat_k_org(1+shift:end);
error = v_log(throw:end) - yhat_k;
error_org = v(throw:end) - yhat_k_org;
var(error)
var(error_shifted)
var(error_org)
var(error_org_shifted)

% Original domain plot (not shifted)
figure()
hold on
plot(yhat_k_org,'g');
plot(v(throw:end));
hold off
basicPlot(error_org,noLags,'Original domain not shifted')

% Original domain plot (shifted)
figure()
hold on
plot(yhat_k_org(1+shift:end),'g');
plot(v(throw:end-shift));
hold off
basicPlot(error_org,noLags,'Original domain')

% VETY UNSURE!!! Lets compare it to the theoretical variance (VERY UNSURE OF THIS- not
% even normal distributed)
theoretical_variance = sum(Fk.^2) * var_1
conf = 2*sqrt(theoretical_variance);
conf_int = [0-conf, 0+conf]
error_outside = (sum(error_org>conf_int(2)) + sum(error_org<conf_int(1)))/length(error_org)

%% Create naive model 
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
