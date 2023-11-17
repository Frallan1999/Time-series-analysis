%
% Time series analysis
% Computer exercise 1  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%%  2.1 Working with time series in Matlab
% create A and C polynomials for ARMA process 
A1 = [1 -1.79 0.84];
C1 = [1 -0.18 -0.11];
A2 = [1 -1.79];
C2 = [1 -0.18 -0.11];

% Create ARMA polynomials 1 and 2 
arma_1 = idpoly(A1, [], C1);
arma_2 = idpoly(A2, [], C2);

arma_1.a            % way of fetching a polynomial 
pzmap(arma_1)       % view poles and zeros of arma

% Simulation of ARMA 
rng(0);             % set seed
sigma2 = 1;         % variance of error terms
N = 100;              % length of resulting vector

e = sqrt(sigma2) * randn(N,1);       % generate normal distributied noice, 
y = filter(arma_1.c, arma_1.a, e);   % simulating an ARMA process

% NOTE: always simulate longer process than needed when simulating a process
% containing an AR part, and then omitt the initial samples. Prefer ab 
% exaggerated number of omitted samples. Assume inital effects will be
% negligible after say 100 samples 

% Now we simulate using our created function "simulateMyARMA" 
y_hat = simulateMyARMA(arma_1.c, arma_1.a, sigma2, N);

% Now we want to simulate new versions
N = 300; 
sigma2 = 1.5; 

y1 = simulateMyARMA(arma_1.c, arma_1.a, sigma2, N);
y2 = simulateMyARMA(arma_2.c, arma_2.a, sigma2, N);

%% Question 1

figure(1)
subplot(211)
plot(y1)
subplot(212)
plot(y2)

figure(2)
subplot(211)
pzmap(arma_1)
subplot(212)
pzmap(arma_2)

% we can see that y2 process diverges. When studying the poles and zeros for
% that arma, we see a pole outside the unit circle. 

%% Question 2
% Information about covariance.
% Theoretical: The "kovarians" function in matlab can be used to calculate 
% the theoretical covariance function r_y(k) for an arma process. Function 
% assumes that the driving noise process has unit variance, i.e. V(et) =
% sigma2 = 1. 
% Estimated: use the function r_est = covf(y,m)

% finding theoretical and estimated covariance for arma_1
m = 20;         %m is the maximum lag value
r_theo = kovarians(arma_1.c, arma_1.a, m);      % caluclate theoretical covariance function
stem(0:m, r_theo*sigma2);
hold on
r_est = covf( y1, m+1 )         % calculate estimated covariance function
stem(0:m, r_est, 'r');

% Question: Why are the estimated and theoretical covariance functions not
% identical?  
% Answer: .... 

%Remember that for the estimated ACF we should only use lags up to N/4. 
              
%% Question 3

% call on function that does basic analysis by plotting the acf, pacf, 
% and normplot of your data. 

figure(3)
basicPlot(y,m,'Test')

% from these plots we choose the model order
na = 2;
nc = 2; %For now

data = iddata(y1);          % make data an object type for estimation

% We now estimate an ar and arma based on our choice of order: 
ar_model = arx(data, na);  % est model using LS method arx for AR(na) 
arma_model = armax(data, [na nc]);    % est model using MS method for ARMA(na, ca)

% Display the estimated parameters, their std and model FPE (final prediction error) 
present(ar_model)
present(arma_model)

% Calculate error residual of estimated model, note that we switched places
% of a and c polynomials to get the inverse

%We wonder what to send in here - should it be poly and just y? 
e_hat = filter (arma_model.a, arma_model.c, y);

figure(4) %How do we see that they are corrupted?
plot(e_hat(1:20))

%Remove the na first error estimations. We create a separate model for
%this, myFilter. 
e_hat = e_hat(length(arma_model.a):end);

%% Question 3 contd. testing AR models

basicPlot(y1,m,'Data y1')

%Testing AR(2)
na = 2;
ar_model = arx(y1, na);  
e_hat_ar = myFilter(ar_model.a, ar_model.c, y1);
basicPlot(e_hat_ar,m,'AR(2)');
present(ar_model)
%Looks good, FPE: 1.518

%Testing AR(3), bit unsure of third coeff in first plot
na = 3;
ar_model = arx( y1, na);  
e_hat_ar = myFilter(ar_model.a, ar_model.c, y1);
basicPlot(e_hat_ar,m,'AR(3)');
present(ar_model)
%Looks good, FPE: 1.516. Slightly lower, and third param significant
%(slightly). However, smaller model is better and this is not SO much
%better. 

%Testing AR(4)
na = 4;
ar_model = arx( y1, na);  
e_hat_ar = myFilter(ar_model.a, ar_model.c, y1);
basicPlot(e_hat_ar,m,'AR(4)');
present(ar_model)
% The 4th is not significant, also higher FPE. 

%% Question 3 contd. testing ARMA models - we see a bit of ringing in both ACF and PACF

% Testing ARMA(2,1)
na = 2;
nc = 1; 
arma_model = armax( y1, [na nc]);    
e_hat_arma = myFilter(arma_model.a, arma_model.c, y1);
basicPlot(e_hat_arma, m, 'ARMA(2,1)')
present(arma_model)
%Worse FPE here than in the AR models, 1.632. 

% Testing ARMA(2,2) as we know it is the true. 
na = 2;
nc = 2; 
arma_model = armax( y1, [na nc]);    
e_hat_arma = myFilter(arma_model.a, arma_model.c, y1);
basicPlot(e_hat_arma, m, 'ARMA(2,2)')
present(arma_model)
% Worse FPE here than in the AR models, 1.632. All significant coeff, still
% some ringing left.

% Testing ARMA(3,2) as we think AR(3) was good. 
na = 2;
nc = 3; 
arma_model = armax( y1, [na nc]);    
e_hat_arma = myFilter(arma_model.a, arma_model.c, y1);
basicPlot(e_hat_arma, m, 'ARMA(3,2)')
present(arma_model)
