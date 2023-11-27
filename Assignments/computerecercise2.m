%
% Time series analysis
% Computer exercise 2  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%% Generate some data following the Box-Jenkins model:

rng(0)
n = 500;                % Number of samples
A3=[1 .5];
C3 = [1 -.3 .2];
w= sqrt(2)*randn(n+100,1); 
x = filter (C3,A3,w);           % Create the input

A1 = [1 -.65];
A2 = [1 .90 .78];
C=1;
B=[0000.4];
e = sqrt(1.5) * randn(n + 100,1);
y = filter(C,A1,e) + filter(B,A2,x); % Create the output

x = x(101:end) , y = y(101:end) % Omit initial samples 
clear A1, A2, C, B, e, w, A3, C3

%% Plotting the data
close all
clc

m=50;
plot(x)
basicPlot(x,m,'x-data') 

%% Create an ARMA-model for the input xt as function of white noise
close all
clc

m=50;

%Initial guess 
%ACF ringing, PACF peak at 1 --> AR(1)?
na = 1;
nc = 0;
data = iddata(x);                   
model_1 = armax(data, [na nc]); 
present(model_1);
e_hat = myFilter(model_1.a, model_1.c, x);
basicPlot(e_hat,m,'ARMA(1,0)')

whitenessTest(e_hat)

%Knowing it's an ARMA(1,2), test it with an ARMA(1,2) too 
%Model 1 according to instructions
na = 1;
nc = 2;
data = iddata(x);                   
model_1 = armax(data, [na nc]); 
present(model_1);
e_hat = myFilter(model_1.a, model_1.c, x);
basicPlot(e_hat,m,'ARMA(1,2)')

whitenessTest(e_hat)

%We tested the same thing with idpoly and got the same result
% Seems that it's nice to use when wanting to control what orders are
% "active" in the polynomial

%% Pre-whitening