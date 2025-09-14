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
load proj23.mat
%% 2.1 Studying the rain data for Kassala
close all; 

% Saving data in new variables
rain_org = Kassala.rain_org;
rain_org_t = Kassala.rain_org_t;
rain = Kassala.rain;
rain_t = Kassala.rain_t;

figure
plot(rain_t,rain)
figure
plot(ElGeneina.rain_t, ElGeneina.rain)


%% 2.1: Gaussian analysis of original rain data
% Are outliers a problem? 

nbrLags = 50;
subplot(121)
acf(rain_org, nbrLags, 0.02 ,1);
hold on
tacf(rain_org, nbrLags, 0.02, 0.02 ,1);
hold off
title('ACF and TACF with alpha=0.02')

subplot(122)
acf(rain_org, nbrLags, 0.01 ,1);
hold on
tacf(rain_org, nbrLags, 0.0, 0.01 ,1);
hold off
title('ACF and TACF with alpha=0.01')

%% 2.1: Gaussian analysis of original rain data
% Identifying outliers
close all
clc

checkIfNormal(rain_org,'Rain','D',0.05) %Observations 344, 175, 56, 464 seem to be a bit outside

o1 = 344;
o2 = 175;
o3 = 56;
o4 = 464;

figure(2) % They are the extreme values
hold on
plot(rain_org_t,rain_org);
plot(rain_org_t(o1),rain_org(o1),'r*');
plot(rain_org_t(o2),rain_org(o2),'r*');
plot(rain_org_t(o3),rain_org(o3),'r*');
plot(rain_org_t(o4),rain_org(o4),'r*');
legend('data','637','639','454','201');
hold off

%% 2.1: Gaussian analysis of original rain data
% Substituting the outliers
close all
clc

rain_org_e = Kassala.rain_org;
rain_org_e(o1) = (rain_org_e(o1-1) + rain_org_e(o1+1))*1/2;
rain_org_e(o2) = (rain_org_e(o2-1) + rain_org_e(o2+1))*1/2;
rain_org_e(o3) = (rain_org_e(o3-1) + rain_org_e(o3+1))*1/2;
rain_org_e(o4) = (rain_org_e(o4-1) + rain_org_e(o4+1))*1/2;

plot(rain_org_e);

%% 2.1: Gaussian analysis of original rain data
% Did it make a difference? Nope!
close all
clc

nbrLags = 50;
subplot(121)
acf(rain_org_e, nbrLags, 0.02 ,1);
hold on
tacf(rain_org_e, nbrLags, 0.02, 0.02 ,1);
hold off
title('ACF and TACF with alpha=0.02')

subplot(122)
acf(rain_org_e, nbrLags, 0.01 ,1);
hold on
tacf(rain_org_e, nbrLags, 0.01, 0.01 ,1);
hold off
title('ACF and TACF with alpha=0.01')

%% 2.1: Gaussian analysis of original rain data
close all; 

nbrLags = 50;
figure(1)
plot(rain_org_t, rain_org)
title('rain org')
subplot(121)
lambda_max = bcNormPlot(rain_org,1)
title('Box Cox plot of rain org')

fprintf('The Box-Cox curve is maximized at %4.2f. This is very close to zero, and thus suggests that a log-transform might be helpful.\n', lambda_max)
subplot(122)
normplot(rain_org)
checkIfNormal(rain_org, 'Kassala rain org');

% Looking at the Normal probability plot. The rain_org data does not look gaussian at all.
% Looking at the BJ curve we see a maximization close to zero -> suggesting
% a log transform might be helpful

%% 2.1: Gaussian analysis of original rain data
% Adding constant and log transforming the data
close all; 

% Log transforming data with constant
constant = 1;    
log_rain_org = log(rain_org + constant);

% Plotting the log_rain_org data
nbrLags = 50;
figure(1)
plot(rain_org_t, log_rain_org)
checkIfNormal(log_rain_org, 'Kassala log rain')

% It is still not Gaussian, but we look away and say yey 

% %% 2.1.2: Finding a reasonable initial a1
% % LOG DATA
% % We want to model our rain as an AR(1) and reconstruct the rain
% % using a Kalman filter. To get an idea of what the a parameter in the
% % AR(1) process could be, we start by trying to model our log_rain_org as an
% % AR(1) to get an idea
% close all; 
% 
% % We do a basic plot
% basicPlot(log_rain_org, nbrLags, 'log rain org')
% % See a lot of seasonality in ACF, disregard this and try to model as AR(1)
% 
% model_init = idpoly([1 0], [], []);
% data = iddata(log_rain_org);
% model_ar = pem(data, model_init);
% present(model_ar)
% res = myFilter(model_ar.c, model_ar.a, log_rain_org);
% basicPlot(res, nbrLags, 'res');

%% 2.2: Finding a reasonable initial a1 
% ORIGINAL DATA
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

%% 2.3: Kalman reconstruction
% Now that we are done with transforming the data and have found an 
% inital estimate for a1, lets go ahead with a Kalman reconstruction. 
close all;
%y = log_rain_org;                               % Redefine the data as y for simplicity        
y = rain_org; 

% Define the state space equations. Value of a1 optimized for ORIGINAL
% (non-log) data.
a1 = 0.172;
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-4 0 0; 1e-6 0 0; 0 1e-6 0];            % try different values
Rw = 1e-4;                                       % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(log_rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1];                               % C_{t | t-1}
    yhat(t) = Ct * xt_t1;                       % y_t{t | t-1} 
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1} (reffered to as y_tilde in project)

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
for k = 1:length(rain_kalman)
    if (rain_kalman(k) < 0)
        rain_kalman(k) = 0;
    end
end

%% 2.3: Kalman reconstruction
% Finding the a1 that minimize the sum error (turned out to be a1 = 0.1720)
close all;

A1 = linspace(0.1,0.9,500);
diff = zeros(length(A1),1);
y = rain_org; 

for i = 1:length(A1)

% Define the state space equations. Value of a1 optimized for ORIGINAL
% (non-log) data.
a1 = A1(i);
A = [a1 0 0; 1 0 0; 0 1 0];    
Re = [1e-4 0 0; 1e-6 0 0; 0 1e-6 0];            % try different values
Rw = 1e-4;                                       % try different values

% Set some initial values
xt_t1 = [0 0 0]';                               % Initial state values for rain denser time scale
Rxx_1 = 10 * eye(3);                            % Initial state variance: large V0 --> small trust in initial values

% Vectors to store values in
N = length(log_rain_org);
Xsave = zeros(3,N);                             % Stored states: We have three hidden states (a1 is assumed known)
ehat = zeros(3,N);                              % Prediction residual (??? is this right) 

for t=1:N
    Ct = [1 1 1];                               % C_{t | t-1}
    yhat(t) = Ct * xt_t1;                       % y_t{t | t-1} 
    ehat(t) = y(t) - yhat(t);                   % e_t = y_t - y_{t | t-1} (reffered to as y_tilde in project)

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
for k = 1:length(rain_kalman)
    if (rain_kalman(k) < 0)
        rain_kalman(k) = 0;
    end
end

diff(i) = abs(sum(rain_kalman) - sum(rain_org));

end 

[mindiff, minindex] = min(diff);
a1 = A1(minindex);
plot(A1, diff)

%% 2.3: Kalman reconstruction
% New time scale for Kalman reconstruction 
clc

rain_kalman_t = zeros(length(rain_org_t)*3,1);
year_difference = 10/(30*12);                       % 10 days between in years 

for t = 1:length(rain_kalman_t) 
    if mod(t,3) == 0
        rain_kalman_t(t-2) = rain_org_t(t/3)-year_difference*2;
        rain_kalman_t(t-1) = rain_org_t(t/3)-year_difference;
        rain_kalman_t(t) = rain_org_t(t/3);
    end
end

%% 2.3: Kalman reconstruction
% ORIGINAL DATA
% Plotting the results
figure(1);
subplot(311);
plot(rain_kalman_t, rain_kalman)
title('Reconstructed rain')
subplot(312);
plot(rain_org_t, rain_org)
title('Original rain measurements')
subplot(313);
plot(rain_t, rain)
title('Interpolated rain')

sum(rain_kalman)                 
sum(rain_org)                               
abs(sum(rain_kalman)-sum(rain_org))

save('Kassala_kalman.mat', 'rain_kalman', 'rain_kalman_t');
