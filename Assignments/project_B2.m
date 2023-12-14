%
% Time series analysis
% Assignment 

clear; 
close all;
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)

%% 3. Modeling and validation for El-Geneina
load proj23.mat

% Split the data
% n = length(ElGeneina.nvdi);

model = ElGeneina.nvdi(1:453,1);         % 70% for modelling
m_t = ElGeneina.nvdi_t(1:453,1);

valid = ElGeneina.nvdi(454:584,1);       % 20% for validation
v_t = ElGeneina.nvdi_t(454:584,1);

test = ElGeneina.nvdi(584:end,1);        % 10% for test
t_t = ElGeneina.nvdi_t(584:end,1); 

max_data = 255;
min_data = 0;

m = 2*(model-min_data)/(max_data - min_data)-1;
v = 2*(valid-min_data)/(max_data - min_data)-1;
t = 2*(test-min_data)/(max_data - min_data)-1;

% Transformation of data 

m_log = log(m);
v_log = log(v);
t_log = log(t);

plot(m_t, model);
nbrLags = 50;

%% Dividing the (input) rain data into the same modeling, validation and test data as (output) NVDI 
% Dividing the x-data 
clc
close all

load rain_kalman.mat

figure(1)
subplot(2,1,1)
plot(rain_kalman_t, rain_kalman)          
title('Reconstructed rain in Kalman')
subplot(2,1,2)
plot(ElGeneina.rain_org_t, ElGeneina.rain_org) 
title('Rain without interpolation')
  
x_all = rain_kalman;            % From A, the reconstructed rain in original domain
rain_t = rain_kalman_t;         % From A, the time line is slightly different with reconstructed rain than interpolated. Adjusted for this. 
xm_t = rain_t(1:1245);          % Ensure both input and output model data ends same date (1994). 

% The time for model x will be longer than modeling y, as we have rain data from long before we have NVDI data. 
% Validation and test data should cover same time periods for both.
xm = x_all(1:length(xm_t));

% Does rain REALLY matter 20 years in advance? We experience with making it
% shorter to easier identify true delay between x and y
xm_short = x_all(700:length(xm_t));
xm_t_short = rain_t(700:1245);

xv = x_all(length(xm)+1:length(xm)+length(v_log));
xv_t = v_t;                 % Should be the exact same time

xt = x_all(length(xm)+length(xv):end);
xt_t = t_t;

figure(2)
subplot(4,1,1)
plot(rain_kalman_t, rain_kalman)
title('Full reconstructed rain')
subplot(4,1,2)
plot(xm_t, xm)
title('Modeling data, x')
subplot(4,1,3)
plot(v_t, xv)
title('Validation data, x')
subplot(4,1,4)
plot(t_t, xt)
title('Test data, x') % Seems correct

%% Fit Box-Jenkins to the data
% Transform input data x for easier modeling 
clc
close all

% We wish to model yt = B(z)z^-d / A2(z) * xt + C1(z) / A1(z) * et

% First, attempt to find transfer function H(z) = B(z) z^-d / A2(z) 
% from xt to yt
% We know xt is not white, thus we need to perform pre-whitening
% Form an ARMA model of the input A3(z) xt = C3(z) wt
% We may need to make x more Gaussian in order to fit an ARMA. (?)

bcNormPlot(xm) % Suggesting log transformation might be useful 
xm_log = log(xm+1);
xm_short_log = log(xm_short+1);
basicPlot(xm_short_log, nbrLags, 'X model data')
checkIfNormal(xm_short_log,'X model data');

%% Fit Box-Jenkins to the data 
% Pre-whitening step: Fitting an ARMA to A3 x = C3 et
clc
close all

y = m_log;
%x = xm_short_log;
x = xm_log(end-length(y)+1:end); %We want to start at the same time in order to get 

% our version before
% A3 = [1 0 0];
% C3 = [1 zeros(1,35) -1];

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

%% Fit Box-Jenkins to the data 
% Compute CCF eps and w, eps_t = H(z) * w_t + v_t
close all;

% Replace xt with wt and pre-whiten y to form eps = H(z)wt + vt
eps_t = myFilter(c3a3.a, c3a3.c, y);
w_t = myFilter(c3a3.a, c3a3.c, x);

basicPlot(eps_t,50,'Eps_t');
basicPlot(w_t,50,'W_t');

n = length(x); 

M=100;
figure()
stem(-M:M,crosscorr(w_t ,eps_t,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% r (A2 order): exponential decay suggests 0
% d (delay for B): could be 2
% s (order for B after delay): 1 

%% Fit Box-Jenkins to the data 
% Testing model orders for A2 and B
d = 4;                                      % To be updated
% A2 = [1 0 0]; 
% B = [zeros(1,7) 1 zeros(1,5)];
A2 = 1;
B = [0 0 0 0 1 0];

Mi = idpoly ([1] ,[B] ,[] ,[] ,[A2]);
z = iddata(y,xm_log(end-d-length(y)+1:end-d));    % Length adjusted to delay 
ba2 = pem(z,Mi); present(ba2)
etilde = resid (ba2, z );

%% Fit Box-Jenkins to the data 
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

% Etilde doesn't have to be white --> we have only modeled half of BJ
% Next step is to derive orders of C1 and A1 from etilde = C1 / A1 et -->
% ARMA 

%% Fit Box-Jenkins to the data 
% Determine orders for A1 and C1, model etilde = C1/A1 * e
close all;
clc; 
basicPlot(etilde.y,nbrLags,'etilde, look for A1 and C1');

A1 = [1 0 zeros(1,34) -1] ; 
C1 = 1;
model_init = idpoly (A1, [], C1);
model_init.Structure.A.Free = [0 1 zeros(1,34) 1];

etilde_data = iddata(etilde.y);
a1c1 = pem(etilde_data,model_init); 
present(a1c1)
res_tilde = resid (a1c1, etilde_data);
basicPlot(res_tilde.y,nbrLags,'ARMA(1,0)');
whitenessTest(res_tilde.y);

%% Fit Box-Jenkins to the data 
% Reestimate the full model with pem (as model will change when we do it all together) 
close all;
clc; 

%B = [zeros(1,7) 1 zeros(1,3)];
%A2 = [1 0 0];
Mi = idpoly(1, B, C1, A1, A2);
Mi.Structure.D.Free = [0 1 zeros(1,34) 1];
%Mi.Structure.B.Free = [zeros(1,10) 1];
%Mi.Structure.F.Free = [0 0 1];
MboxJ = pem(z,Mi);
present(MboxJ)
ehat = resid(MboxJ,z);      % the estimate of the noise process e_t

%% Fit Box-Jenkins to the data 
% Final analysis of ehat
% ehat and x should be uncorrelated
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

%% Predict the input, x



%% Use the Box-Jenkins model for NVDI prediction





