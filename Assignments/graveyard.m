%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)
%% B1 model prediction 
hold on
plot(yhat_k(1+shiftK:end));         % blue
plot(v_log(throw:end-shiftK));
hold off
basicPlot(error,noLags,'new')


%% TEST WITH NABLA INSTEAD

% Model the data after differentiating 
close all; 
clc;
noLags = 50; 

% initial model - try a1, a36, c36
A = [1 zeros(1,36)];
C = [1 zeros(1,36)];
data_m_diff_mean = iddata(m_diff_mean);

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.a.Free = [0 1 zeros(1,34) 1];
model_init.Structure.c.Free = [zeros(1,36) 1];
model_ar1 = pem(data_m_diff_mean, model_init);       % Optimize variables 
res = resid(model_ar1, data_m_diff_mean);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_ar1);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(1)');
plotNTdist(res.y);
% Looks white, but residual not normal --> can't 100 procent trust result.
% the residual is however t-dstriuted, even wider confidence interval for
% testing whiteness, so it is OK! 

%% TEST WITH NABLA INSTEAD incorp diff

% Model the data after differentiating 
close all; 
clc;
noLags = 50; 

% initial model - try a1, a36, c36
A = [1 zeros(1,36)];
C = [1];
data_m_log = iddata(m_log);

model_init = idpoly(A, [], C);                       % Set up initial model
model_init.Structure.a.Free = [0 1 zeros(1,34) 1];
model_ar1 = pem(data_m_log, model_init);       % Optimize variables 
res = resid(model_ar1, data_m_log);            % Calculate residuals 
plotACFnPACF(res.y, noLags, "Residual for AR(1) modelling");
figure()
present(model_ar1);
whitenessTest(res.y);
checkIfNormal(res.y,'Residuals for AR(1)');
plotNTdist(res.y);
% Looks white, but residual not normal --> can't 100 procent trust result.
% the residual is however t-dstriuted, even wider confidence interval for
% testing whiteness, so it is OK! 

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

%% KLADD! Do we like negative rain? NO -> one option is to put to zero, other to move up? 
% here lets try putting it to zero :) 
close all;

rain_kalman_pos = zeros(length(rain_kalman),1);
for t=1:length(rain_kalman_pos)
    if rain_kalman(t) < 0
        rain_kalman_pos(t) = 0;
    else 
        rain_kalman_pos(t) = rain_kalman(t);
    end
end

figure(1);
subplot(311);
hold on
plot(rain_t, rain_kalman_pos)
scatter(rain_t, rain_kalman_pos)
hold off
subplot(312);
hold on
plot(rain_org_t, log_rain_org)
scatter(rain_org_t, log_rain_org)
hold off
subplot(313);
hold on
plot(rain_t, log(rain+constant))
scatter(rain_t, log_rain)
hold off

sum(rain_kalman_pos)
sum(log_rain_org)

%% Graveyard
% Examine if any points diverge from the distribution
figure(2)
%checkIfNormal(nvdi,'NVDI','D',0.05) %Observations 637, 639, 454, 201 seem to be a bit outside

% Plotting the potential outliers - they are a bit outside
figure(2)
hold on
plot(nvdi_t,nvdi);
plot(nvdi_t(637),nvdi(637),'r*');
plot(nvdi_t(639),nvdi(639),'g*');
plot(nvdi_t(454),nvdi(454),'r*');
plot(nvdi_t(201),nvdi(201),'g*');
legend('data','637','639','454','201');
hold off

%% Removing the outliers and substituting them with the interpolated version
nvdi_outlier = ElGeneina.nvdi;

nvdi_outlier(637) = 1/2 * (nvdi_outlier(638) + nvdi_outlier(636));
nvdi_outlier(639) = 1/2 * (nvdi_outlier(640) + nvdi_outlier(638));
nvdi_outlier(454) = 1/2 * (nvdi_outlier(453) + nvdi_outlier(455));
nvdi_outlier(201) = 1/2 * (nvdi_outlier(202) + nvdi_outlier(200));

figure(3)
plot(nvdi_t, nvdi_outlier)

checkIfNormal(nvdi_outlier,'NVDI','D',0.05) %Observations 637, 639, 454, 201 seem to be a bit outside

% Normalizing NVDI data as given in the assignment