%
% Time series analysis
% Mini project 1  
%
%
clear; 
close all;
addpath('../functions', '../data')     % Add this line to update the path

% Stores signal in y and the signalss sampling rate is stored in Fs
[y,Fs] = audioread('fa.wav');

%% Task 1: Plot and sound of signal 
plot(y)
sound(y);

%% Task 2: Extract 200 samples from vowel
% choose first vowel y
startPoint = 2800;
endPoint =  startPoint+199;
vowel = y(startPoint:endPoint);

% Plotting realisation of vowel. See real-valued periodic signal 
figure 
plot(vowel)

% plotting the estimated correlation function (as we can also see in the realisation)
% 95% confidence interval
% From this we see a repetence in the plot and that will give us the
% periodicity, and taking 1 over this periodicity we get the frequency 
noLags = 80; 
significance = 0.05;
figure
acf(vowel, noLags, significance, 1)         % Not sure what plotIt = 1 entails? 
title('Correlation-domain')

%% 
% test test test
