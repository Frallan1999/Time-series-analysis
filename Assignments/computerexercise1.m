%
% Time series analysis
% Computer exercise 1  
%
%
clear; 
close all;
addpath('../functions', '../data')     % Add this line to update the path

%%  2.1 Working with time series in Matlab
% create A and C polynomials for ARMA process 
A1 = [1 -1.79 0.84];
C1 = [1 -0.18 -0.11];
A2 = [1 -1.79];
C2 = [1 -0.18 -0.11];

% Create ARMA polynomials 1 and 2 
ARMA_1 = idpoly(A1, [], C1);
ARMA_2 = idpoly(A2, [], C2);
