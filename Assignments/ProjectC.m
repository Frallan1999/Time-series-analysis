%
% Time series analysis
% Assignment 

clear; 
close all; 
clc; 
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path
%% 4. Data from before 
load proj23.mat

model = ElGeneina.nvdi(1:453,1);         % 70% for modelling
m_t = ElGeneina.nvdi_t(1:453,1);

valid = ElGeneina.nvdi(454:584,1);       % 20% for validation
v_t = ElGeneina.nvdi_t(454:584,1);

test = ElGeneina.nvdi(584:end,1);        % 10% for test
t_t = ElGeneina.nvdi_t(584:end,1); 

max_data = 255;
min_data = 0;

m = 2*(model-min_data)/(max_data - min_data)-1;
v = 2*(valid-min_data)/(max_data - min_data)-1;
t = 2*(test-min_data)/(max_data - min_data)-1;

% Transformation of data 
m_log = log(m);
v_log = log(v);
t_log = log(t);

plot(m_t, model);
nbrLags = 50;

%% 4. Data from before 
% Dividing the (input) rain data into the same modeling, validation and test data as (output) NVDI 
clc
close all

load rain_kalman.mat;

figure(1)
subplot(2,1,1)
plot(rain_kalman_t, rain_kalman)          
title('Reconstructed rain in Kalman')
subplot(2,1,2)
plot(ElGeneina.rain_org_t, ElGeneina.rain_org) 
title('Original rain data')
  
x_all = rain_kalman;            % From A, the reconstructed rain
rain_t = rain_kalman_t;         % From A, the reconstructed rain timeline 
xm_t = rain_t(1:1245);          % Ensure both input and output model-data ends same date (1994). 

% The time for model x will be longer than modeling y, as we have rain data from long before we have NVDI data. 
xm = x_all(1:length(xm_t));

% Make model rain data for same period as model NVDI data 
% (true modelling data) 
xm_real = xm(end-length(m_t)+1:end);
xm_real_t = m_t; 

% Validation and test data should cover same time periods for both.
xv = x_all(length(xm)+1:length(xm)+length(v_log));
xv_t = v_t;                 % Should be the exact same time

xt = x_all(length(xm)+length(xv):end);
xt_t = t_t;

figure(2)
subplot(4,1,1)
plot(rain_kalman_t, rain_kalman)
title('Full reconstructed rain')
subplot(4,1,2)
plot(xm_t, xm)
title('Modeling data, x')
subplot(4,1,3)
plot(v_t, xv)
title('Validation data, x')
subplot(4,1,4)
plot(t_t, xt)
title('Test data, x') % Seems correct

%% 4. (from before) 
clc
close all

bcNormPlot(x_all)                                      % -> Log transformation might be useful 
constant = 1; 
xm_log = log(xm + constant);             
xm_real_log = log(xm_real + constant);                       % Used!
xv_log = log(xv + constant);
xt_log = log(xt + constant);

basicPlot(xm_real_log, nbrLags, 'X model data')
checkIfNormal(xm_real_log,'X model data');
%% 4.1 Recursive update of input predictions
% Create Kalman for recursive estimation 
load input_arma.mat
input_arma = c3a3;
close all;
clc; 

% Data to put into Kalman
y_input = [xm_real_log;xv_log;xt_log];
N = length(y_input);      % all data??? Is this really correct 

% Prediction step and number unknownd
k = 9;                                  % Prediction step 
q0 = nnz(input_arma.a) - 1              % (3) - Number of unknowns in the A polynomial 
p0 = nnz(input_arma.c) - 1              % (4) - Number of unknowns in the C polynomial 

% Define the state space equations.
A = eye(p0+q0);                         % p0 + q0 are number of hidden states 
Rw = 0.2;                               % Try different! Measurement noise covariance matrix. Same dimension as Ry.
Re = 1e-6*eye(p0+q0);                   % Try different! System noise covariance matrix. 

% Set initial values
xt_t = [-0.1626 -0.2716 -0.5327 0.1276 0.3341 -0.2683 0.2605]';    
Rxt_t1 = 10*eye(p0+q0);                 % Initial covariance matrix, IF large -> small trust initial values 

% Storing values 
Xsave = zeros(p0+q0,N-k);               % Stored (hidden) states
ehat = zeros(1,N);                      % One-step prediction residual
yhat = zeros(N,1);                      % Estimated output ({yhat}_{t|t-1}) (NOT one step prediction)  
yhatk = zeros(N,1);                     % Estimated k-step prediction

y_t_input = zeros(1,N);                 

xStd  = zeros(p0+q0,N-k);               % Stores one std for the one-step prediction.
xStdk = zeros(p0+q0,N-k);               % Stores one std for the k-step prediction.

for t=37:N-k                            % Starts at 37 as we use t-36
    % Update the predicted state and the time-varying state vector.
    xt_t1 = A*xt_t;                     % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ -y_input(t-1) -y_input(t-2) -y_input(t-36) ehat(t-1) ehat(t-2) ehat(t-6) ehat(t-9)];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ryt_t1 = Ct * Rxt_t1 * Ct' + Rw;    % R^yy_{t | t-1} = C R^xx_{t|t-1} + Rw
    Kt = Rxt_t1 * Ct' / Ryt_t1;         % K_t = R^xx{t| t-1} * Ct' * inv(Ryy{t | t-1})
    yhat(t) = Ct*x_t1;                  % One step prediction - y{t|t-1}
    ehat(t) = y_input(t)-yhat(t);       % One step prediction error - e_t = y_t - y_{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);          % x_{t | t}
    
    % Update the covariance matrix estimates
    Rxt_t  = Rxt_t1 - Kt*Ryt_t1*Kt';    % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rxt1_t = A*Rxt_t*A' + Re;           % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    y_t_input(1:t) = y_input(1:t);      % Store all values known at t 

    % Form redictions 
    for k0=1:k
        Ck = [ -y_t_input(t-1+k0) -y_t_input(t-2+k0) -y_t_input(t-36+k0) ehat(t+k0-1) ehat(t+k0-2) ehat(t+k0-6) ehat(t+k0-9)]; % C_{t+k|t}
        y_t_input(t+k0) = Ck*A^k0*xt_t;  % \{yhat}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
    end

    yhatk(t+k) = y_input_pred(t+k0);  
    Xsave(:,t) = xt_t;


    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );            % This is one std for each of the parameters for the k-step prediction.
end

% Change to original domain!
yhatk_org = exp(yhat) - constant; 
% error = real

%% Show the one-step prediction. 
figure
plot( [y_input(1:N-k) yhat] )
title('One-step prediction using the Kalman filter')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
e1 = y_input(200:N-k)-yhat(200:end);                   % Ignore the initial values to let the filter converge first.
plotACFnPACF( e1, 40, 'One-step prediction using the Kalman filter');
fprintf('Examining the one-step residual.\n')
checkIfWhite( e1 );

%% Show the k-step prediction. 
figure
plot( [y_input yhatk] )
title( sprintf('%i-step prediction using the Kalman filter', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])

%% Examine k-step prediction residual.
ek = y_input(N-200:N-k)-yhatk(N-200:N-k);             % Ignore the initial values to let the filter converge first.
plotACFnPACF( ek, 40, sprintf('%i-step prediction using the Kalman filter', k)  );

fprintf('  The variance of original signal is                %5.2f.\n', var(y_input)')
fprintf('  The variance of the 1-step prediction residual is %5.2f.\n', var(e1)')
fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(ek)')

%% Simulate data for testing 
A_sim = input_arma.a;
C_sim = input_arma.c;
