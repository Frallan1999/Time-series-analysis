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

%% Question 2

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

% we can see that y2 proess diverges. When studying the poles and zeros for
% that arma, we see a pole outside the unit circle. 

%% Question 2
% Information about covariance.
% Theoretical: The "kovarians" function in matlab can be used to calculate 
% the theoretical covariance function r_y(k) for an arma process. Function 
% assumes that the driving noise process has unit variance, i.e. V(et) =
% sigma2 = 1. 
% Estimated: use the function r_est = covf(y,m)

% finding theoretical and estimated covariance for arma_1
m = 20; 
r_theo = kovarians(arma_1.c, arma_1.a, m);      % caluclate theoretical covariance function
stem(0:m, r_theo*sigma2);
hold on
r_est = covf( y1, m+1 )         % calculate estimated covariance function
stem(0:m, r_est, 'r');

% Question: Why are the estimated and theoretical covariance functions not
% identical?  
% Answer: .... 
              
%% Question 3

% call on function that does basic analysis by plotting the acf, pacf, 
% and normplot of your data. 

data = iddata(y1);          % make data an object type for estimation

ar_model = arx( y1, [na]);  % est model using LS method arx for AR(na) 
arma_model = armax( y1, [na nc])    % est model using MS method for ARMA(na, ca) 