%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%%
%% 2.1 (Instead of Log)  If we don't interpret lambda max as zero, test with the transformation instead
rain_org_bc = (rain_org_c.^lambda_max - 1) / lambda_max

% Plotting the transformed data
nbrLags = 50;
figure(1)
plot(rain_org_t, rain_org_bc)
% checkIfNormal(rain_org_bc , 'ElGeneina rain_org')

%%
%% Does the sum add up? 

y_fake = zeros(length(Xsave),1);
for i = 1:length(Xsave)
    y_fake(i) = sum(Xsave(:,i));
end

figure(1)
plot(y_fake)
figure(2)
plot(log_rain_org