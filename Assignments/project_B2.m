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
y_all = ElGeneina.nvdi;
y_t = ElGeneina.nvdi_t;

max_data = 255;
min_data = 0;

y_all = 2*(y_all-min_data)/(max_data - min_data)-1;

% Split it 
ym = y_all(1:453,1);         % 70% for modelling
m_t = y_t(1:453,1);

yv = y_all(454:584,1);       % 20% for validation
v_t = y_t(454:584,1);

yt = y_all(585:end,1);        % 10% for test
t_t = y_t(585:end,1); 

% Take the log
y_log = log(y_all);
ym_log = log(ym);
yv_log = log(yv);
yt_log = log(yt);

% Useful for prediction
ym_yv = [ym; yv]; 
ym_yv_t = [m_t; v_t];
ym_yv_log = log(ym_yv);

modelLim = length(ym)+1; % Index for first data in validation set

%% 3.3 Model B2 - NVDI prediction with rain as external input
% Examine the rain data - is it Gaussian?
close all; 
clc; 

load rain_kalman.mat;
x_all = rain_kalman; 
x_t = rain_kalman_t;         % From A, the reconstructed rain timeline 

checkIfNormal(x_all,'Reconstructed rain','D',0.05)
bcNormPlot(x_all) % Suggests that taking the log might be a good idea

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
xv_t = v_t;                 % Should be the exact same time

xt = x_all(end-length(t_t)+1:end);
xt_t = t_t;

% Making a vector with modeling AND validation data, for prediction use
xm_xv = [xm; xv];
xm_xv_t = [xm_t; xv_t];
modelLim = length(xm)+1; % Index for the first validation data point
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

%% 3.3.1 Prewhitening of the precipitation data
clc
close all

% We wish to model yt = B(z)z^-d / A2(z) * xt + C1(z) / A1(z) * et
% First, attempt to find transfer function H(z) = B(z) z^-d / A2(z) 
% from xt to yt
% We know xt is not white, thus we need to perform pre-whitening
% Form an ARMA model of the input A3(z) xt = C3(z) wt
% Fitting an ARMA to A3 x = C3 et
clc
close all

x = xm_log;
nbrLags = 50;
A3 = [1 zeros(1,35) -1];
C3 = [1 zeros(1,35) -1];
model_init = idpoly(A3 ,[], C3);
model_init.Structure.a.Free = [0 1 1 zeros(1,33) 1];
model_init.Structure.c.Free = [0 zeros(1,35) 1];

% Test this
% A3 = [1 zeros(1,35) -1];
% C3 = [1 zeros(1,9)];
% model_init = idpoly(A3 ,[], C3);
% model_init.Structure.a.Free = [0 1 1 zeros(1,7) zeros(1,25) 0 1];
% model_init.Structure.c.Free = [0 1 1 zeros(1,3) 1 0 0 1];

c3a3 = pem(x, model_init);

res = resid(c3a3, x); 
present(c3a3);
basicPlot(res.y,nbrLags,'Residuals')
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for ARMA, prewhitening');
% plotNTdist(res.y);

%% 3.3.2 Box Jenkins 
% Compute CCF eps and w, eps_t = H(z) * w_t + v_t
close all;

y = ym_log;

% Simulate wt and eps_t
eps_t = myFilter(c3a3.a, c3a3.c, y);
w_t = myFilter(c3a3.a, c3a3.c, x);

%basicPlot(eps_t,50,'Eps_t');
%basicPlot(w_t,50,'W_t');

n = length(x); 

M=100;
figure()
stem(-M:M,crosscorr(w_t ,eps_t,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% r (A2 order): ringing suggests 2
% d (delay for B): 2
% s (order for B after delay): 2

%% 3.3.2 Box Jenkins 
% Testing model orders for A2 and B                               
A2 = [1 0 0]; 
B =  [0 0 1 0];     % Removed insignificant 2nd order 

Mi = idpoly ([1] ,[B] ,[] ,[] ,[A2]);
z = iddata(y,x);    
Mba2 = pem(z,Mi); 
present(Mba2)
etilde = resid (Mba2, z);

%% 3.3.2 Box Jenkins
% Check if etilde = yt - Bz^d / A2 xt is uncorrelated with xt - it should be 
close all;
clc; 

M=40;
stem(-M:M,crosscorr(etilde.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% Looks quite horrible, however we will move on
% Next step is to derive orders of C1 and A1 from etilde = C1 / A1 et -->
% ARMA 

%% 3.3.2 Box Jenkins
% Determine orders for A1 and C1, model etilde = C1/A1 * e
close all;
clc; 
basicPlot(etilde.y,nbrLags,'etilde, look for A1 and C1');

A1 = [1 zeros(1,35) -1] ; 
C1 = [1 0 1];
model_init = idpoly (A1, [], C1);
model_init.Structure.a.Free = [0 1 zeros(1,34) 1];
model_init.Structure.c.Free = [0 0 1];

etilde_data = iddata(etilde.y);
a1c1 = pem(etilde_data,model_init); 
present(a1c1)
res_tilde = resid (a1c1, etilde_data);
basicPlot(res_tilde.y,nbrLags,'ARMA(A1,C1)');
checkIfNormal(res_tilde.y, 'residual e tilde');
whitenessTest(res_tilde.y);

% White enough, moving on

%% 3.3.2 Box Jenkins
% Reestimate the full model with pem (as model will change when we do it all together) 
close all;
clc; 

%AMANDA's EDITS
C1 = 1;         % Removal of insignificant variables
% END

Mi = idpoly(1, B, C1, A1, A2);
Mi.Structure.D.Free = [0 1 zeros(1,34) 1];  % VA?
%Mi.Structure.C.Free = [0 0 1];
%Mi.Structure.F.Free = [0 0 1];
model_B2 = pem(z,Mi);
present(model_B2)
ehat = resid(model_B2,z);      % the estimate of the noise process e_t

%% 3.3.2 Box Jenkins
% Final analysis of ehat - is the entire model good enough?
close all;
clc; 

M=40;
stem(-M:M,crosscorr(ehat.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

%% Fit Box-Jenkins to the data 
% Final analysis of ehat contd.
% ehat should be white and normally distributed
close all;
clc; 

whitenessTest(ehat.y);
basicPlot(ehat.y,nbrLags,'e-hat');
checkIfNormal(ehat.y,'e-hat','D',0.05);

% The model we will use for prediction is then MboxJ. 

%% 3.3.2 Predict the input
% Fitting an ARMA to A3 x = C3 et
clc
close all

% Finding a good model to the data
x = xm_log;

A3 = [1 zeros(1,35) -1];
C3 = [1 zeros(1,9)];
model_init = idpoly(A3 ,[], C3);
model_init.Structure.a.Free = [0 1 1 zeros(1,7) zeros(1,25) 0 1];
model_init.Structure.c.Free = [0 1 1 zeros(1,3) 1 0 0 1];

c3a3 = pem(x, model_init);

res = resid(c3a3, x); 
present(c3a3);
basicPlot(res.y,nbrLags,'Residuals')
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for ARMA, prewhitening');
% plotNTdist(res.y);

% White? No! But lets move on :) 
save('input_arma.mat','c3a3')

%% 3.3.2 Predict the input
% Using the derived ARMA for predicting the input 

clc
close all

k = 2;                  % sets number of steps prediction
noLags = 50;

% Solve the Diophantine equation and create predictions
[Fx, Gx] = polydiv(c3a3.c, c3a3.a, k);
throw = max(length(Gx), length(c3a3.c));
xhat_k = filter(Gx, c3a3.c, xm_xv_log);
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

figure
acf( ehat, nbrLags, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat );
pacfEst = pacf( ehat, nbrLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

%% 3.3.3 Predicting NVDI with rain as external input
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

[Fy, Gy] = polydiv(KC, KA, k)

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh, Ghh] = polydiv(conv(Fy, KB), KC, k);

% Form the predicted output signal using the predicted input signal.
yhat_k  =  filter(Fhh, 1, xhat_k) + filter(Ghh, KC, xm_xv_log) + filter(Gy, KC, ym_yv_log); 

yhat_k_org = exp(yhat_k);
%(1/2*(exp(yhat_k)+1)*(max_data - min_data))-min_data;
% Inverse transform of 2*(y_all-min_data)/(max_data - min_data)-1;

figure
plot([ym_yv yhat_k_org] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('NVDI', 'Predicted NVDI', 'Prediction starts')
title( sprintf('Predicted output signal, y_{t+%i|t}', k) )
axis([1 length(ym_yv) min(ym_yv)*1.25 max(ym_yv)*1.25])

%% 3.3.3 Predicting NVDI with rain as external input
% Checking the residuals

ehat = ym_yv - yhat_k_org;
ehat = ehat(modelLim:end);

figure
acf(ehat, noLags, 0.05, 1);
title(sprintf('ACF of the %i-step output prediction residual', k) )
checkIfWhite( ehat );
pacfEst = pacf( ehat, noLags, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );
