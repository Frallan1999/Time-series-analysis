%
% Time series analysis
% Computer exercise 1  
%
%
clear; 
close all;
addpath('functions', '/data')     % Add this line to update the path

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

%% Question 1 

                         