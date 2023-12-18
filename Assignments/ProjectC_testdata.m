
%
% Time series analysis
% Assignment 

clear; 
close all; 
clc; 
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path
%% 4. Time-varying model for El-Geneina
% Transforming the NVDI (y) data as in B1

load proj23.mat

% Normalizing the data
y_org = ElGeneina.nvdi;
y_t = ElGeneina.nvdi_t;

max_data = 255;
min_data = 0;

y_all = 2*(y_org-min_data)/(max_data - min_data)-1;

% Split it 
ym = y_all(1:453,1);         % 70% for modelling
m_t = y_t(1:453,1);
ym_org = y_org(1:453,1);

yv = y_all(454:584,1);       % 20% for validation
v_t = y_t(454:584,1);
yv_org = y_org(454:584,1);

yt = y_all(585:end,1);        % 10% for test
t_t = y_t(585:end,1); 
yt_org = y_org(585:end,1); 

% Transformation of data 
y_log = log(y_all);
ym_log = log(ym);
yv_log = log(yv);
yt_log = log(yt);

% Useful for prediction
ym_yv = [ym; yv]; 
ym_yv_t = [m_t; v_t];
ym_yv_log = log(ym_yv);

modelLim = length(ym)+1; % Index for first data in validation set
testlim = length(ym_yv)+1; % Index for first data in test set
%% 4. Time-varying model for El-Geneina
% Loading previous rain data
% Dividing the (input) rain data into the same modeling, validation and test data as (output) NVDI 
clc
close all

load rain_kalman.mat;

load rain_kalman.mat;
x_all = rain_kalman;                     % From A, the reconstructed rain
x_t = rain_kalman_t;                     % From A, the reconstructed rain timeline 
  
constant = 1;
x_log = log(x_all+constant);

% Split it 
xm_long_t = x_t(1:1246);                 % Ensure both input and output model-data ends same date (1994). 
xm_long = x_all(1:length(xm_long_t));
xm = xm_long(end-length(m_t)+1:end);
xm_t = m_t; 

xv = x_all(modelLim:modelLim+length(v_t)-1);
xv_t = v_t;                 

xt = x_all(length(xm)+length(xv):end);
xt_t = t_t;

% Transformation of data
xm_log = log(xm + constant);             
xv_log = log(xv + constant);
xt_log = log(xt + constant);

% Useful for prediction 
xm_xv = [xm; xv];
xm_xv_t = [xm_t; xv_t];
xm_xv_log = log(xm_xv + constant);
modelLim = length(xm)+1;                 % Index for the first validation data point

%% 4. Time-varying model for El-Geneina
% Loading previous model
load input_arma.mat
input_arma = sarima_x;

load("model_B2.mat")
model_B2 = model_B2; 

%% 4.1 Recursive update of rain FINAL
% FINAL recursive model for the rain 
% C1 PARAMETER HAS BEEN REMOVED!!

% Create Kalman for recursive estimation 
close all;
clc; 

% Data to put into Kalman
yx_input = x_log;               % All loged data                   
N = length(yx_input);      

% Prediction step and number unknownd
k = 1;                                  % Prediction step 
q0 = nnz(input_arma.a) - 1              % (3) - Number of unknowns in the A polynomial 
p0 = nnz(input_arma.c) - 1 - 1          % (3) - Number of unknowns in the C polynomial 

% Define the state space equations.
A = eye(p0+q0);                         % p0 + q0 are number of hidden states 
Rw = var(yx_input);                     % Try different! Measurement noise covariance matrix. Same dimension as Ry.
Re = 1e-6*eye(p0+q0);                   % Try different! System noise covariance matrix. 

% Set initial values
xt_t = [-0.1626 -0.2716 -0.5327 0.3462 -0.2665 0.2634]';    
Rxt_t1 = 3*eye(p0+q0);                  % Initial covariance matrix, If large -> small trust initial values 

% Storing values 
Xsave = zeros(p0+q0,N-k);               % Stored (hidden) states
ehat = zeros(1,N);                      % One-step prediction residual
yxhat = zeros(N-k,1);                   % Estimated output ({yhat}_{t|t-1}) (NOT one step prediction)  
yxhatk = zeros(N,1);                    % Estimated k-step prediction

yx_t_input = zeros(1,N);                % Stores all values known at t that can be used        
rain_pred_t = zeros(k,N);               % Stores the t+1|t to t+k|t predictions to be used later in the rows

xStd  = zeros(p0+q0,N-k);               % Stores one std for the one-step prediction.
xStdk = zeros(p0+q0,N-k);               % Stores one std for the k-step prediction.

for t=37:N-k                            % Starts at 37 as we use t-36
    % Update the predicted state and the time-varying state vector.
    xt_t1 = A*xt_t;                     % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ -yx_input(t-1) -yx_input(t-2) -yx_input(t-36) ehat(t-2) ehat(t-6) ehat(t-9)];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ryt_t1 = Ct * Rxt_t1 * Ct' + Rw;    % R^yy_{t | t-1} = C R^xx_{t|t-1} + Rw
    Kt = Rxt_t1 * Ct' / Ryt_t1;         % K_t = R^xx{t| t-1} * Ct' * inv(Ryy{t | t-1})
    yxhat(t) = Ct*xt_t1;                % One step prediction - y{t|t-1}
    ehat(t) = yx_input(t)-yxhat(t);     % One step prediction error - e_t = y_t - y_{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);          % x_{t | t}
    
    % Update the covariance matrix estimates
    Rxt_t  = Rxt_t1 - Kt*Ryt_t1*Kt';    % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rxt_t1 = A*Rxt_t*A' + Re;           % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form k step prediction using the k-1, k-2 and so on prediction
    yx_t_input(1:t) = yx_input(1:t);    % Store all values known at t 
    Rx_k = Rxt_t1;

    for k0=1:k
        Ck = [ -yx_t_input(t-1+k0) -yx_t_input(t-2+k0) -yx_t_input(t-36+k0) ehat(t+k0-2) ehat(t+k0-6) ehat(t+k0-9)]; % C_{t+k|t}
        yxk = Ck*A^k*xt_t;              % \{yhat}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        yx_t_input(t+k0) = yxk; 
        Rx_k = A*Rx_k*A' + Re;
        rain_pred_t(k0,t+k0) = yxk;     % For later use 
    end

    yxhatk(t+k) = yxk;  
    Xsave(:,t) = xt_t;

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rxt_t) );    % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );    % This is one std for each of the parameters for the k-step prediction.
end

%% 4.2 Recursive update of NVDI - model C FINAL
% THIS IS THE FINAL KAMLAN FILTER
% Parameters excluded: b4, b3, b5, a38, c1, b40, a2, and a37
close all;
clc;

KA_kalman = [-0.68 -0.13 -0.21];
KB_kalman = [0.02];
KC_kalman = [0.11]; 

% Data to put into Kalman
yx_input = x_log;       % all data
y_input = y_log;        % all data
N = length(y_input); 

% Prediction step and number unknownd
k = k;                                  % OBS! Code is set up in way that k here MUST be same as above! (even though it we only need two less in prediction) 
nbr_params = length(KA_kalman) + length(KB_kalman) + length(KC_kalman);

% Define the state space equations.
A = eye(nbr_params);                    % Hidden states matrix 
Rw = var(y_input);                      % Try different! Measurement noise covariance matrix. Could use from noise estimate polynomial pred
Re = 1e-10*eye(nbr_params);             % Try different! System noise covariance matrix. 

% Set initial values
xt_t = [KA_kalman KB_kalman KC_kalman]';    
Rxt_t1 = 3*eye(nbr_params);             % Initial covariance matrix, IF large -> small trust initial values 

% Storing values 
Xsave = zeros(nbr_params,N-k);          % Stored (hidden) states
ehat = zeros(1,N);                      % One-step prediction residual
yhat = zeros(N-k,1);                    % Estimated output ({yhat}_{t|t-1}) (NOT one step prediction)  
yhatk = zeros(N,1);                     % Estimated k-step prediction

y_t_input = zeros(1,N);          
yx_t_input2 = zeros(1,N);                 

xStd  = zeros(nbr_params,N-k);          % Stores one std for the one-step prediction.
xStdk = zeros(nbr_params,N-k);          % Stores one std for the k-step prediction.

for t=41:N-k                            % Starts at 37 as we use t-36
    % Update the predicted state and the time-varying state vector.
    xt_t1 = A*xt_t;                     % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ -y_input(t-1)  -y_input(t-3) -y_input(t-36) yx_input(t-39) ehat(t-2) ];
    
    % Update the parameter estimates.
    Ryt_t1 = Ct * Rxt_t1 * Ct' + Rw;    % R^yy_{t | t-1} = C R^xx_{t|t-1} + Rw
    Kt = Rxt_t1 * Ct' / Ryt_t1;         % K_t = R^xx{t| t-1} * Ct' * inv(Ryy{t | t-1})
    yhat(t) = Ct*xt_t1;                 % One step prediction - y{t|t-1}
    ehat(t) = y_input(t)-yhat(t);       % One step prediction error - e_t = y_t - y_{t | t-1}
    xt_t = xt_t1 + Kt*ehat(t);          % x_{t | t}
   
    % Update the covariance matrix estimates
    Rxt_t  = Rxt_t1 - Kt*Ryt_t1*Kt';    % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rxt_t1 = A*Rxt_t*A' + Re;           % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form k step prediction using the k-1, k-2 and so on prediction
    y_t_input(1:t) = y_input(1:t);      % Store all values known at t 
    yx_t_input2(1:t) = yx_input(1:t);   % Store all values known at t
    yx_t_input2(t+1:t+k) = rain_pred_t(:,t);
    Rx_k = Rxt_t1;

    for k0=1:k
        Ck = [ -y_t_input(t+k0-1)  -y_t_input(t+k0-3) -y_t_input(t+k0-36) yx_t_input2(t+k0-39) ehat(t+k0-2)];
        yk = Ck*A^k*xt_t;               % \{yhat}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        y_t_input(t+k0) = yk ;  
        Rx_k = A*Rx_k*A' + Re;
    end

    yhatk(t+k) = yk;  
    Xsave(:,t) = xt_t;

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rxt_t) );    % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );    % This is one std for each of the parameters for the k-step prediction.
end

%% 4.2 Recursive update of NVDI - model C
% Examine the estimated parameters (for simulated data) 
close all;
figure()
% trueParams = [KA_kalman KB_kalman KC_kalman];     
% plotWithConf( (1:N-k), Xsave', xStd', trueParams);
plotWithConf( (1:N-k), Xsave', xStd');

title('Estimated states for output data nvdi')
xlim([42 N-k])

figure()
plot(Xsave(:,42:end)')
xline(testlim-50, 'r--', 'LineWidth', 1, 'Label','');
legend('a1', 'a3', 'a36', 'b39', 'c2')
xlim([1, length(Xsave)-50]);

fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(Xsave(:,1))
    fprintf('Estimated value: %5.2f (+/- %5.4f).\n', Xsave(k0,end), xStd(k0,end) )
end 

%% 4.2 Recursive update of NVDI - model C
% Examine k-step prediction residual.close all; 

y_input_org = y_org;
yhatk_org = exp(yhatk);
yhatk_org = 1/2*(yhatk_org+1)*(max_data - min_data)+min_data;
 
error = y_input_org(testlim+40:end)-yhatk_org(testlim+40:end);        % Steady state around 40 after 

mean_error = abs(mean(error))
plotACFnPACF( error, 20, sprintf('%i-step prediction using the Kalman filter', k)  );
errorM = y_input_org(40:testlim-1)-yhatk_org(40:testlim-1); 
errorMV = [error' errorM']';

fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(error)')
fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(errorMV)')
fprintf('  The normalized variance of the %i-step prediction residual is %5.2f.\n', k, var(error)/var(yt_org(40:end)))