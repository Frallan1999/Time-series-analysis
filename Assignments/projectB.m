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
% Normalizing the data

y_org = ElGeneina.nvdi;
y_t = ElGeneina.nvdi_t;

max_data = 255;
min_data = 0;

y = 2*(y_org-min_data)/(max_data - min_data)-1;

figure
plot(y_t,y)
title('Normalized data')                % Data deemed stationary enough

figure
plot(y_t, ElGeneina.nvdi)
title('Original data')

%% 3.1 Dividing and cleaning of data set
% Examine the data's stationarity 
% Split into modelling, validation and test data

% Split the data
n = length(ElGeneina.nvdi);

ym = y(1:453,1);         % 70% for modelling
m_t = y_t(1:453,1);
ym_org = y_org(1:453,1);

yv = y(454:584,1);       % 20% for validation
v_t = y_t(454:584,1);
vm_org = y_org(454:584,1);

yt = y(585:end,1);        % 10% for test
t_t = y_t(585:end,1); 
yt_org = y_org(585:end,1); 

% Plot it
figure(2)
subplot(311)
plot(m_t,ym);
title('Modeling set')
subplot(312)
plot(v_t,yv);
title('Validation set')
subplot(313)
plot(t_t,yt);
title('Test set')

%% 3.1 Dividing and cleaning of data set
% Examine outliers in modeling set -> looks fine
close all
clc

nbrLags = 50;
subplot(121)
acf(ym, nbrLags, 0.02 ,1);
hold on
tacf(ym, nbrLags, 0.02, 0.02 ,1);
hold off
title('ACF and TACF with alpha=0.02')

subplot(122)
acf(ym, nbrLags, 0.01 ,1);
hold on
tacf(ym, nbrLags, 0.01, 0.01 ,1);
hold off
title('ACF and TACF with alpha=0.01')
%% 3.2 Model B1 - NVDI prediction without external input
% Checking need for transformation
close all; 
clc; 

checkIfNormal(y,'normalized NVDI','D',0.05)
%bcNormPlot(y) % Suggests that taking the log might be a good idea

%% 3.2 Model B1 - NVDI prediction without external input
% Testing log transform
close all; 
clc; 

y_log = log(y);
checkIfNormal(y_log,'log of normalized NVDI','D',0.05) % Did the trick!

%%  3.2 Model B1 - NVDI prediction without external input
% Transformation of data 
close all; 
clc; 

ym_log = log(ym);
yv_log = log(yv);
yt_log = log(yt);
ym_yv = [ym; yv]; % Useful for prediction
ym_yv_t = [m_t; v_t];
ym_yv_log = log(ym_yv);
modelLim = length(ym)+1; % Index for first data in validation set
testlim = length(ym_yv)+1;
plot(ym_yv_t, ym_yv)

%%  3.2 Model B1 - NVDI prediction without external input
% Differentiation of the data; 
clc; 
close all; 
noLags = 50;                % max up to N/4

plotACFnPACF(ym_log,noLags, 'model data');

% Differentiate on season 36 with nabla (1-z^-36)
A36 = [1 zeros(1,35) -1];                       % Sets the season
ym_diff = filter(A36,1,ym_log);                   % Filter on seasonality 36 
ym_diff = ym_diff(length(A36):end);               % Omit initial samples
data = iddata(ym_diff);                          % Create object for estimation
figure()
plot(ym_diff);
plotACFnPACF(ym_diff, noLags, "model data after differentiation with nabla36");
mean(ym_diff)

% Also tested "softer" differentiation, to better result but non-zero mean:

% Differentiate on season 36 with (1-0.35*z^-36)
% A36 = [1 zeros(1,35) -0.35];                     % Sets the season
% m_diff = filter(A36,1,ym_log);                    % Filter on seasonality 36 
% m_diff = m_diff(length(A36):end);                % Omit initial samples
% data = iddata(m_diff);                           % Create object for estimation
% plotACFnPACF(m_diff, noLags, "model data with a36 = 0.35");  
% figure()
% plot(m_diff);
% mean(m_diff)
% 
% % Remove mean of differentiated data
% m_diff_mean = m_diff - mean(m_diff);            % Note! This mean will prob not be same for validation and so on 
% % m_diff_mean = m_diff;
% figure()
% plot(m_diff_mean)

%%  3.2 Model B1 - NVDI prediction without external input
% Model the data after differentiating the data - using differentiation of
% -1 to get zero-mean
close all; 
clc;
noLags = 50; 

% Initial model - try a1
A = [1 0];
C = 1;
ym_diff_d = iddata(ym_diff);

model_init = idpoly(A, [], C);               % Set up initial model
model_ar = pem(ym_diff_d, model_init);       % Optimize variables 
res = resid(model_ar, ym_diff_d);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_ar);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(1)');
plotNTdist(res.y);


% FPE: 0.04411 and Monti: 28.47 < 36.42
% OBS! Residual not normal --> can't 100 procent trust result.
% The residual is however t-dstriuted, even wider confidence interval for
% testing whiteness, so it is OK! 
% ACF & PACF --> Maybe include c36 or/and c3? 

%%  3.2 Model B1 - NVDI prediction without external input
% Although white, a bit of a peak at 36 still -> try a1 & c36

A = [1 0];
C = [1 zeros(1,36)];

model_init = idpoly(A, [], C);                          % Set up initial model
model_init.Structure.c.Free = [0 zeros(1,35) 1];        % added ones C changed
model_arma = pem(ym_diff_d, model_init);                % Optimize variables 
res = resid(model_arma, ym_diff_d);                     % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1 and c36 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1 and c36 modelling');
plotNTdist(res.y);

% FPE: 0.04292 and Monti: 30.74 < 36.42
% All significant, not white residuals

%% 3.2 Model B1 - NVDI prediction without external input
% Although white, a bit of a peak at 36 still -> try a1, c3 & c36
A = [1 0];
C = [1 zeros(1,36)];

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.c.Free = [0 0 0 1 zeros(1,32) 1];        % added ones C changed
model_arma = pem(ym_diff_d, model_init);       % Optimize variables 
res = resid(model_arma, ym_diff_d);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1, c3, and c36 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1, c3, and c36 modelling');
plotNTdist(res.y);
% FPE: 0.04254 and Monti: 28.84 < 36.42
% C3 insignificant --> KISS rule 

%% 3.2 Model B1 - NVDI prediction without external input
% Substituting c36 with c3 only? To make it more simple

A = [1 0];
C = [1 0 0 1];

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.c.Free = [0 0 0 1];        % added ones C changed
model_arma = pem(ym_diff_d, model_init);       % Optimize variables 
res = resid(model_arma, ym_diff_d);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for a1 and c3 modelling");
figure()
present(model_arma);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for a1 and c3 modelling');
plotNTdist(res.y);
% FPE: 0.04409 and Monti: 22.00 < 36.42
% ACF and PACF shows season of 36, however - more white than ever. We test
% to incorporate season instead and leave c36 out

%%  3.2 Model B1 - NVDI prediction without external input
% Model data without differentiation (incorporating the season)
close all; 
clc; 

noLags = 50; 

% initial model - based on information from differentiated
A = [1 0];
A = conv([1 zeros(1,35) -1], A);
C = [1 0 0 1];

ym_log_d = iddata(ym_log); % Not using differentiated data anymore

model_init = idpoly(A, [], C);
model_init.Structure.a.Free = [0 1 zeros(1,34) 1 0];        
model_init.Structure.c.Free = [zeros(1,3) 1];            
model_B1 = pem(ym_log_d, model_init);                     
res = resid(model_B1, ym_log_d);
plotACFnPACF(res.y, noLags, "Residual for SARIMA model");

figure()
present(model_B1);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for SARIMA model');
plotNTdist(res.y);
% FPE: 0.04658 and Monti: 8.89 < 36.42
% Best one so far
% All significant, however, t-distributed residuals

%% 3.2.2. Model prediction
% Predict the vegetation using the derived final model for VALIDATION DATA
clc
close all

k = 7;                  % sets number of steps prediction
noLags = 50;

% Solve the Diophantine equation and create predictions
[Fx, Gx] = polydiv(model_B1.c, model_B1.a, k);
throw = max(length(Gx), length(model_B1.c));
yhat_k = filter(Gx, model_B1.c, ym_yv_log);

yhat_k_org = exp(yhat_k);
yhat_k_org = 1/2*(yhat_k_org+1)*(max_data - min_data)+min_data;
ym_yv_org = 1/2*(ym_yv+1)*(max_data - min_data)+min_data;

figure
plot(ym_yv_t, [ym_yv_org yhat_k_org] )
line( [ym_yv_t(modelLim) ym_yv_t(modelLim)], [0 200 ], 'Color','red','LineStyle',':' )
legend('NVDI', 'Predicted NVDI', 'Prediction starts')
title( sprintf('Predicted NVDI, validation data, y_{t+%i|t}', k) )
axis([ym_yv_t(length(ym)) ym_yv_t(end) min(ym_yv_org)*0.9 max(ym_yv_org)*1.1])

% figure
% plot([ym_yv_org yhat_k_org] )
% %line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
% %legend('NVDI', 'Predicted NVDI', 'Prediction starts')
% title( sprintf('Predicted NVDI, y_{t+%i|t}', k) )
% %axis([length(ym) length(ym_yv_org) min(ym_yv_org)*0.9 max(ym_yv_org)*1.1])

%% 3.2.2 Model prediction
% Form the residual for the validation data. It should behave as an MA(k-1)

ehat = ym_yv_org - yhat_k_org;
ehat = ehat(modelLim:end);
var_ehat = var(ehat)
var_ehat_norm = var(ehat)/var(yv_org)

figure
acf( ehat, nbrLags, 0.05, 1 );
title( sprintf('ACF of the %i-step prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat );
pacfEst = pacf( ehat, nbrLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%% 3.2.2. Model prediction
% Predict the vegetation using the derived final model for TEST DATA
clc
close all

k = 7;                  % sets number of steps prediction
noLags = 50;

% Solve the Diophantine equation and create predictions
[Fx, Gx] = polydiv(model_B1.c, model_B1.a, k);
throw = max(length(Gx), length(model_B1.c));
yhat_k = filter(Gx, model_B1.c, y_log);

yhat_k_org = exp(yhat_k);
yhat_k_org = 1/2*(yhat_k_org+1)*(max_data - min_data)+min_data;

figure
plot(y_t, [y_org yhat_k_org] )
line( [y_t(testlim) y_t(testlim)], [0 200 ], 'Color','red','LineStyle',':' )
legend('NVDI', 'Predicted NVDI', 'Prediction starts')
title( sprintf('Predicted NVDI, test data, y_{t+%i|t}', k) )
axis([y_t(length(ym_yv)) y_t(end) min(ym_yv_org)*0.9 max(ym_yv_org)*1.1])

%% 3.2.2 Model prediction
% Form the residual for the TEST data. It should behave as an MA(k-1)

ehat = y_org - yhat_k_org;
ehat = ehat(testlim:end);
var_ehat = var(ehat)
var_ehat_norm = var(ehat)/var(yt_org)

figure
acf( ehat, nbrLags, 0.05, 1 );
title( sprintf('ACF of the %i-step prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat );
pacfEst = pacf( ehat, nbrLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );
