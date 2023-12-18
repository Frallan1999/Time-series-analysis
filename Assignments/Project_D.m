%
% Time series analysis
% Assignment 

clear; 
close all; 
clc; 
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)

%% 3.3 Model B2 - NVDI prediction with rain as external input
% Transforming the NVDI (y) data as in B1

load proj23.mat

% Normalizing the data
y_org = Kassala.nvdi;
y_t = Kassala.nvdi_t;

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
%% 3.3 Model B2 - NVDI prediction with rain as external input
% Examine the rain data - is it Gaussian?
close all; 
clc; 

load Kassala_kalman.mat;
x_all = rain_kalman; 
x_t = rain_kalman_t;         % From A, the reconstructed rain timeline 

checkIfNormal(x_all,'Reconstructed rain','D',0.05)
%bcNormPlot(x_all) % Suggests that taking the log might be a good idea


%% 3.3 Model B2 - NVDI prediction with rain as external input
% Log transformation

constant = 1;
x_log = log(x_all+constant);
checkIfNormal(x_log,'Log rain','D',0.05) % Not perfect, but a lot better

%% 3.3 Model B2 - NVDI prediction with rain as external input
% Dividing the (input) rain data into the same modeling, validation and test data as (output) NVDI 
% Dividing the x-data 
clc
close all

figure(1)
subplot(2,1,1)
plot(rain_kalman_t, rain_kalman)          
title('Reconstructed rain in Kalman')
subplot(2,1,2)
plot(ElGeneina.rain_org_t, ElGeneina.rain_org) 
title('Original rain data')
 
xm_long_t = x_t(1:1246);          % Ensure both input and output model-data ends same date (1994). 

% The time for model x will be longer than modeling y, as we have rain data from long before we have NVDI data. 
xm_long = x_all(1:length(xm_long_t));

% Make model rain data for same period as model NVDI data
xm = xm_long(end-length(m_t)+1:end);
xm_t = m_t; 

% Validation and test data should cover same time periods for both.
xv = x_all(modelLim:modelLim+length(v_t)-1);
xv_t = v_t;                

xt = x_all(end-length(t_t)+1:end);
xt_t = t_t;

% Making a vector with modeling AND validation data, for prediction use
xm_xv = [xm; xv];
xm_xv_t = [xm_t; xv_t];
modelLim = length(xm)+1; % Index for the first validation data point
xm_xv_xt = [xm; xv; xt];

figure(2)
plot(xm_xv_t, xm_xv);
title('Modeling and validation data')

figure(3)
subplot(4,1,1)
plot(rain_kalman_t, rain_kalman)
title('Full reconstructed rain')
subplot(4,1,2)
plot(m_t, xm)
title('Modeling data, x')
subplot(4,1,3)
plot(v_t, xv)
title('Validation data, x')
subplot(4,1,4)
plot(t_t, xt)
title('Test data, x') % Seems correct

%% 3.3 Model B2 - NVDI prediction with rain as external input
% Taking the log of the different parts
clc
close all

xm_log = log(xm + constant);             
xv_log = log(xv + constant);
xt_log = log(xt + constant);
xm_xv_log = log(xm_xv + constant);


%% TEST WITH THE MODEL

load model_B2.mat
load input_arma.mat

%% 3.3.2 Predict the input
% Using the derived ARMA for predicting the input 
clc
close all

k = 7;                  % sets number of steps prediction
noLags = 50;

% Solve the Diophantine equation and create predictions
[Fx, Gx] = polydiv(sarima_x.c, sarima_x.a, k);
throw = max(length(Gx), length(sarima_x.c));
xhat_k = filter(Gx, sarima_x.c, xm_xv_log);
xhat_k_org = exp(xhat_k)-constant;

figure
plot([xm_xv xhat_k_org] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('Reconstructed input rain', 'Predicted rain', 'Prediction starts')
title( sprintf('Predicted input signal, x_{t+%i|t}', k) )
axis([1 length(xm_xv) min(xm_xv)*1.25 max(xm_xv)*1.25])

%std_xk = sqrt( sum( Fx.^2 )*var_ex );
%fprintf( 'The theoretical std of the %i-step prediction error is %4.2f.\n', k, std_xk)

%% 3.3.2 Predict the input
% Form the residual for the validation data. It should behave as an MA(k-1)
ehat = xm_xv - xhat_k_org;
ehat = ehat(modelLim:end);
var_ehat = var(ehat)
var_ehat_norm = var(ehat)/var(xv)

figure
acf( ehat, noLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%% 3.3.3 Predicting NVDI with rain as external input
% VALIDATION DATA
% We have the model on the form y = C/A1 et + B/A2 * xt_t and need it on
% ARMAX form in order to make predictions
% A1 A2 yt = A2 C1 et + A1 B z^-d x

% Let KA yt = KC et + KB xt form an ARMA process 
% KA = A1 A2 (F * D in idpoly --> A in ARMA) 
% KB = A1 B (D * B in idpoly --> B in ARMA)
% KC = A2 C1 (F * C in idpoly --> C in ARMA)

clc 
close all

KA = conv(model_B2.F,model_B2.D);
KB = conv(model_B2.D, model_B2.B);
KC = conv(model_B2.F,model_B2.C);

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for x_t, this is for y_t).

[Fy, Gy] = polydiv(KC, KA, k);

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh, Ghh] = polydiv(conv(Fy, KB), KC, k);

% Form the predicted output signal using the predicted input signal.
yhat_k  =  filter(Fhh, 1, xhat_k) + filter(Ghh, KC, xm_xv_log) + filter(Gy, KC, ym_yv_log); 

yhat_k_org = exp(yhat_k);
yhat_k_org = 1/2*(yhat_k_org+1)*(max_data - min_data)+min_data;
ym_yv_org = 1/2*(ym_yv+1)*(max_data - min_data)+min_data;

figure
plot(ym_yv_t, [ym_yv_org yhat_k_org] )
line( [ym_yv_t(modelLim) ym_yv_t(modelLim)], [0 200 ], 'Color','red','LineStyle',':' )
legend('NVDI', 'Predicted NVDI', 'Prediction starts')
title( sprintf('Predicted NVDI, validation data, y_{t+%i|t}', k) )
axis([ym_yv_t(length(ym)) ym_yv_t(end) min(ym_yv_org)*0.9 max(ym_yv_org)*1.1])

%% 3.3.3 Predicting NVDI with rain as external input
% Checking the residuals

ehat = ym_yv_org - yhat_k_org;
ehat = ehat(modelLim:end);
var_ehat = var(ehat)
var_ehat_norm = var(ehat)/var(yv_org)

noLags = 50;
figure
acf(ehat, noLags, 0.05, 1);
title(sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );


