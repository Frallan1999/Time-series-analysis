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

%% TEST (CAN BE REMOVED)
close all; 
test = ElGeneina.nvdi;
rain_test = rain_kalman(end-length(test)+1:end);
rain_t_test = rain_kalman_t(end-length(test)+1:end);

plot(rain_t_test, rain_test);
figure()
plot(ElGeneina.nvdi_t, test);
figure()
[correlation, lag] = xcorr(rain_test, test);
plot(lag, correlation);

%% Amanda's A

%
% Time series analysis
% Assignment 
%
%
clear; 
close all;
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path (Hanna)
%% 1. Introduction to the data
clear
close all
clc

load proj23.mat
%% 2.1.1 Studying the rain data for El-Geneina
close all; 

% Saving data in new variables
rain_org = ElGeneina.rain_org;
rain_org_t = ElGeneina.rain_org_t;
rain = ElGeneina.rain;
rain_t = ElGeneina.rain_t;

%% 2.1.1: Gaussian analysis of original rain data
nbrLags = 50;
figure(1)
plot(rain_org_t, rain_org)
title('rain org')
figure(2)
lambda_max = bcNormPlot(rain_org,1)
title('Box Cox plot of rain org')

fprintf('The Box-Cox curve is maximized at %4.2f. This is very close to zero, and thus suggests that a log-transform might be helpful.\n', lambda_max)
checkIfNormal(rain_org, 'ElGeneina rain org')

% Looking at the Normal probability plot. The rain_org data does not look gaussian at all.
% Looking at the BJ curve we see a maximization close to zeo -> suggesting
% a log transform might be helpful

%% 2.1.1: Gaussian analysis of original rain data
% Adding constant and log transforming the data
close all; 

% Adding constant to data 
constant = 1;
rain_org_c = rain_org + constant;

% Log transforming data with constant
log_rain_org = log(rain_org_c);

% Plotting the log_rain_org data
nbrLags = 50;
figure(1)
plot(rain_org_t, log_rain_org)
checkIfNormal(log_rain_org, 'ElGeneina rain_org')

% It is still not Gaussian, but we look away and say yey 
%% 2.1.1: Gaussian analysis of original rain data
% Removing the mean 
log_rain_org_m  = log_rain_org - mean(log_rain_org);

% Plotting the log_rain_org data
nbrLags = 50;
figure(3)
plot(rain_org_t, log_rain_org_m)
checkIfNormal(log_rain_org_m, 'ElGeneina rain_org') 
%% 2.1.2: Finding a reasonable initial a1
% We want to model our rain as an AR(1) and reconstruct the rain
% using a Kalman filter. To get an idea of what the a parameter in the
% AR(1) process could be, we start by trying to model our log_rain_org as an
% AR(1) to get an idea
close all; 

% We do a basic plot
basicPlot(rain_org, nbrLags, 'rain org')
% See a lot of seasonality in ACF, disregard this and try to model as AR(1)

model_init = idpoly([1 0], [], []);
data = iddata(rain_org);
model_ar = pem(data, model_init);
present(model_ar)
res = myFilter(model_ar.c, model_ar.a, rain_org);
basicPlot(res, nbrLags, 'res');

%% 2.1.3: Kalman reconstruction - testing values for a 
% Now that we are done with transforming the data and have found an 
% inital estimate for a1, lets go ahead with a Kalman reconstruction. 

A1 = linspace(0.1,0.9,500);
diff = zeros(length(A1),1);


for i = 1:length(A1)

close all;
y = rain_org;                               % Redefine the data as y for simplicity 

% Define the state space equations.
a1 = A1(i);
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-4 0 0; 1e-6 0 0; 0 1e-6 0];           % try different values
Rw = 1e-4;                                       % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1];                               % C_{t | t-1}
    yhat(t) = Ct * xt_t1;                       % y_t{t | t-1} 
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw;                % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy;                     % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);                  % x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1;              % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t;                           % x_{t+1 | t} this is our AR(1) process 
    Rxx_1 = A * Rxx * A' + Re;                  % R^{xx}_{t+1 | t}
       
    Xsave(:,t) = xt_t;
end

% We would like to store this in an vector as in the interpolated case 
rain_kalman = zeros(3*length(rain_org),1); 
for k = 1:length(Xsave)
    rain_kalman(3*k) = Xsave(1,k);
    rain_kalman(3*k-1) = Xsave(2,k);
    rain_kalman(3*k-2) = Xsave(3,k);
end

%No negative rain!
for q = 1:length(rain_kalman)
    if (rain_kalman(q) < 0)
        rain_kalman(q) = 0;
    end
end

diff(i) = abs(sum(rain_kalman) - sum(rain_org));

end

[mindiff, minindex] = min(diff);
a1 = A1(minindex);

%% Plot the minimum a1

figure(1)
plot(A1,diff)
title('Difference in sum between real data and kalman hidden states over different choices of a1')

%% 2.1.3: Kalman reconstruction - True values
% Now that we are done with transforming the data and have found an 
% inital estimate for a1, lets go ahead with a Kalman reconstruction. 

close all;
y = rain_org;                               % Redefine the data as y for simplicity 

% Define the state space equations.
a1 = 0.25;
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-4 0 0; 1e-6 0 0; 0 1e-6 0];           % try different values
Rw = 1e-4;                                       % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1];                               % C_{t | t-1}
    yhat(t) = Ct * xt_t1;                       % y_t{t | t-1} 
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1}

    % Update
    Ryy = Ct * Rxx_1 * Ct' + Rw;                % R^{yy}_{t | t-1}
    Kt = Rxx_1 * Ct' / Ryy;                     % K_t = Rxx{t| t-1} * Ct' * Ryy{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);                  % x_{t | t}
    Rxx = Rxx_1 - Kt * Ct * Rxx_1;              % R^{xx}_{t | t}

    % Predict the next state
    xt_t1 = A * xt_t;                           % x_{t+1 | t} this is our AR(1) process 
    Rxx_1 = A * Rxx * A' + Re;                  % R^{xx}_{t+1 | t}
       
    Xsave(:,t) = xt_t;
end

% We would like to store this in an vector as in the interpolated case 
rain_kalman = zeros(3*length(rain_org),1); 
for k = 1:length(Xsave(1,:))
    rain_kalman(3*k) = Xsave(1,k);
    rain_kalman(3*k-1) = Xsave(2,k);
    rain_kalman(3*k-2) = Xsave(3,k);
end

for i = 1:length(rain_kalman)
    if (rain_kalman(i)<0)
        rain_kalman(i) = 0;
    end
end
%% 2.1.3: Kalman reconstruction
% Plotting the results
figure(1);
subplot(311);
plot(rain_t, rain_kalman)
subplot(312);
plot(rain_org_t, rain_org)
subplot(313);
plot(rain_t, rain)

sum(rain_kalman)                 % relevant if not removed mean 
sum(rain_org)                               % relevant if not removed mean
abs(sum(rain_kalman)-sum(rain_org))

%% 2.1.3: Kalman reconstruction - Simulating data to test the filter
% Generate the hidden states x_t+1 = a1 * x_t + et

N1 = 3*N;
extraN = 100;
A1 = [1 -a1]; 
e = randn(N1+extraN,1); 
x_sim = filter(1, A1, e); x_sim = x_sim(extraN+1:end);

for i = 1:N1
    if(x_sim(i)<0) 
        x_sim(i) = 0;
    end
end

y_sim = zeros(N,1);
v = randn(N,1);

for i = 1:N
    y_sim(i) = x_sim(3*i) + x_sim(3*i-1) + x_sim(3*i-2) + v(i);
end

%% Plot simulation vs reality

figure(1)
plot(rain_kalman)
figure(2)
plot(x_sim)

sum(rain_kalman)
sum(x_sim)
sum(v)

%% Hannas prediction av x på B2 (med mina comments / Amanda)
clc
close all

k = 2;                  % sets number of steps prediction
noLags = 50;

% Solve the Diophantine equation and create predictions
[Fx, Gx] = polydiv(c3a3.c, c3a3.a, k);
throw = max(length(Gx), length(c3a3.c));
xhat_k = filter(Gx, c3a3.c, xm_xv_log); % <-- Andreas använder hela sitt dataset, M och V i vårt fall. 
%xhat_k = xhat_k(throw:end); % <-- Han kastar inte heller några samples :O

% Transform prediction into original domain
xhat_org = exp(xhat_k);

% Create errors 
%error = xv(throw:end) - xhat_org;
error = xv - xhat_org;
var(error)

% Plot in original domain
figure()
hold on
plot(xhat_org,'b');
plot(xv);
%plot(xv(throw:end));
hold off
basicPlot(error,noLags,'Errors') % 

%% Continued

% Transform prediction into original domain
xhat_org = exp(xhat_k)+constant;

% Create errors 
%error = xv(throw:end) - xhat_org;
error = xv - xhat_org(modelLim:end);
var(error)

% Plot in original domain
figure()
hold on
plot(xhat_org,'b');
plot(xv);
%plot(xv(throw:end));
hold off
basicPlot(error,noLags,'Errors') % 

%% HANNA ->
%% 3.2.2. Model prediction
% Lets do a one step and save the variance
k = 1;      % prediction step 

[Fk, Gk] = polydiv(model_sarima.c, model_sarima.a, k );  
throw = max(length(Gk), length(model_sarima.c));
yhat_1 = filter(Gk, model_sarima.c, v_log);                             
yhat_1 = yhat_1(throw:end);   
yhat_1_org = exp(yhat_1);
var_1 = var(v(throw:end) - yhat_1_org)
var_test = var(v_log(throw:end) - yhat_1)

basicPlot(e, 50, 'e')

e = v(throw:end) - yhat_1_org;          % Calculating our error terms that we expect to behave like an MA
eps = myFilter(Fk, 1, e);               % Removing the MA-part, to see if white. 
basicPlot(eps, 50, 'Stationary prediction error') % It is!


%%  3.2.2 Model prediction (B1) 
close all; 
clc; 
k = 45;                  % sets number of steps prediction

% Solve the Diophantine equation and create predictions
[Fk, Gk] = polydiv(model_sarima.c, model_sarima.a, k);
throw = max(length(Gk), length(model_sarima.c));
yhat_k = filter(Gk, model_sarima.c, v_log);
yhat_k = yhat_k(throw:end);

% Transform prediction into original domain
yhat_k_org = exp(yhat_k);

% It can be seen that the shift is IN GENERAL this (and is fun to then
% incorporate to be able to plot for both shifted and non shifted
if k == 1 || k == 2
    shift = k; 
else 
    shift = 3; 
end

% Create the errors (shifted and unshifted, original domain vs not) 
error_shifted = v_log(throw:end-shift) - yhat_k(1+shift:end);
error_org_shifted = v(throw:end-shift) - yhat_k_org(1+shift:end);
error = v_log(throw:end) - yhat_k;
error_org = v(throw:end) - yhat_k_org;
var(error)
var(error_shifted)
var(error_org)
var(error_org_shifted)

% Original domain plot (not shifted)
figure()
hold on
plot(yhat_k_org,'g');
plot(v(throw:end));
hold off
basicPlot(error_org,noLags,'Original domain not shifted')

% Original domain plot (shifted)
figure()
hold on
plot(yhat_k_org(1+shift:end),'g');
plot(v(throw:end-shift));
hold off
basicPlot(error_org,noLags,'Original domain')

% VETY UNSURE!!! Lets compare it to the theoretical variance (VERY UNSURE OF THIS- not
% even normal distributed)
theoretical_variance = sum(Fk.^2) * var_1
conf = 2*sqrt(theoretical_variance);
conf_int = [0-conf, 0+conf]
error_outside = (sum(error_org>conf_int(2)) + sum(error_org<conf_int(1)))/length(error_org)

%% Test naive model on validation data
close all; 
clc; 
k = 37;                  % sets number of steps prediction
% very bad with k less than 37!!!

% Solve the Diophantine equation and create predictions
[Fk, Gk] = polydiv(model_naive.c, model_naive.a, k);
throw = max(length(Gk), length(model_naive.c));
yhat_k = filter(Gk, model_naive.c, v);
yhat_k = yhat_k(throw:end);

% It can be seen that the shift is IN GENERAL this (and is fun to then
% incorporate to be able to plot for both shifted and non shifted
if k == 1 || k == 2
    shift = k; 
else 
    shift = 3; 
end

% Create the errors (shifted and unshifted, original domain vs not) 
error_org_shifted = v(throw:end-shift) - yhat_k(1+shift:end);
error_org = v(throw:end) - yhat_k;
var(error_org)
var(error_org_shifted)

% Original domain plot (not shifted)
figure()
hold on
plot(yhat_k,'g');
plot(v(throw:end));
hold off
basicPlot(error_org,noLags,'Original domain not shifted')

% Original domain plot (shifted)
figure()
hold on
plot(yhat_k(1+shift:end),'g');
plot(v(throw:end-shift));
hold off
basicPlot(error_org,noLags,'Original domain')
