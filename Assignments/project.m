%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)
%% Understanding the data
clear
close all
clc

load proj23.mat

%% 2.1: Studying the rain (org) data for El-Geneina
close all; 
% We start by plotting the rain_org data
nbrLags = 50;
plot(ElGeneina.rain_org_t, ElGeneina.rain_org)
basicPlot(ElGeneina.rain_org, nbrLags, 'ElGeneina.rain_org')

% It does not seem like AR(1) would be the best fit, but as that is the
% task, we start by modelling the data as an AR(1)




%% (3)

figure(1)
histogram(ElGeneina.rain_org)
figure(2)
histogram(log(ElGeneina.rain_org))
figure(3)
histogram(log(ElGeneina.rain_org)-log(mean(ElGeneina.rain_org)))
%figure(4)
%histogram(sqrt(ElGeneina.rain_org)-sqrt(mean(ElGeneina.rain_org)))
