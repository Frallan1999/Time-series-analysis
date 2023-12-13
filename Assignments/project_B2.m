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

load proj23.mat

% Split the data
n = length(ElGeneina.nvdi);

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

%% Bringing in the input data


%% Fitting a model to current input 
clc
close all

load rain_kalman.mat

figure(1)
subplot(2,1,1)
plot(ElGeneina.rain_t, rain_kalman) %Checking that we have the right data
title('Reconstructed rain in Kalman')
subplot(2,1,2)
plot(ElGeneina.rain_org_t, ElGeneina.rain_org) 
title('Rain without interpolation')

rain_t = ElGeneina.rain_t;
x = rain_kalman; %Remember that this is NOT transformed, AND in the wrong time interval. 
xm_t = rain_t(1:1245); %Manual look-up to avoid getting mismatching dates



% The time for model x will be longer than modeling y, as we have rain data from long before we have NVDI data. 
% Validation and test data should cover same time periods for both.
xm = x(1:length(xm_t));

%Does rain REALLY matter 20 years in advance? We make it shorter for easier
%modelling
xm_short = x(500:length(xm_t));
xm_t_short = rain_t(500:1245);

xv = x(length(xm)+1:length(xm)+length(v_log));

xt = x(length(xm)+length(xv):end);

figure(2)
subplot(4,1,1)
plot(ElGeneina.rain_t, rain_kalman)
title('Full reconstructed rain')
subplot(4,1,2)
plot(xm_t, xm)
title('Modeling data, x')
subplot(4,1,3)
plot(v_t, xv)
title('Validation data, x')
subplot(4,1,4)
plot(t_t, xt)
title('Test data, x') %Seems correct

%% Box-Jenkins: Here we go
clc
close all

% We wish to model yt = B(z)z^-d / A2(z) * xt + C1(z) / A1(z) * et

% First, attempt to find transfer function H(z) = B(z) z^-d / A2(z) 
% from xt to yt
% We know xt is not white, thus we need to perform pre-whitening
% Form an ARMA model of the input A3(z) xt = C3(z) wt
% We have in A explained xt as an AR with a1 = 0.2491. We need to make x
% more Gaussian in order to fit an ARMA. 

xm_log = log(xm+1);
xm_short_log = log(xm_short+1);
basicPlot(xm_short_log, 50, 'X model data')
checkIfNormal(xm_short_log,'X model data');

%% Pre-whitening step: Fitting an ARMA to X
% Note that our y is m_log, and our x is xm_log
y = m_log;
x = xm_short_log;


A3 = [1 0 0];
C3 = [1 zeros(1,35) -1];
%C3 = 1;
model_init = idpoly(A3 ,[], C3);
%model_init.Structure.a.Free = [0 1 1];
model_init.Structure.c.Free = [0 zeros(1,35) 1];
pw = pem(x, model_init);

res = resid(pw, x);
basicPlot(res.y,50,'Residuals')
present(pw);

whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for ARMA, prewhitening');
% plotNTdist(res.y);

% Replace xt with wt and pre-whiten y to form eps = H(z)wt + vt

eps_t = myFilter(pw.a, pw.c, y);
w_t = myFilter(pw.c, pw.a, x);

%% Compute CCF for eps_t = H(z) * w_t + v_t
close all;
M=40;
stem(-M:M,crosscorr(w_t ,eps_t,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% r (A2 order): ringing suggests r = 2 
% d (delay for B): could be 7
% s (order for B after delay): s+d = time of decay, s+d = 12, --> s = 5 

%% Testing the found model orders

A2 = [1 0 0]; 
B = [zeros(1,7) 1 zeros(1,5)];
Mi = idpoly ([1] ,[B] ,[] ,[] ,[A2]);
z = iddata(y,x(1:length(y)));    % fattar att första är data och andra är input 
Mba2 = pem(z,Mi); present(Mba2)
etilde = resid (Mba2, z );

%% Check etilde 
close all;
clc; 

%etilde and x should be uncorrelated --> Looks good, not sign diff from 0
M=40;
stem(-M:M,crosscorr(etilde.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

%% Is etilde white? 
% No, but we have now only modelled half of the BJ model, as y depends on
% something with x AND the error term.   
close all;
clc; 

basicPlot(etilde.y,m,'etilde');
whitenessTest(etilde.y);
figure()

%% 