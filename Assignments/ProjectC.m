%
% Time series analysis
% Assignment 

clear; 
close all; 
clc; 
% addpath('functions', '/data')         % Add this line to update the path
addpath('../functions', '../data')      % Add this line to update the path
%% 4. Time-varying model for El-Geneina
% Loading previous nvdi data
load proj23.mat

y_org = ElGeneina.nvdi;
y_t = ElGeneina.nvdi_t; 

max_data = 255;
min_data = 0;

y = 2*(y_org-min_data)/(max_data - min_data)-1;

% Split it 
n = length(ElGeneina.nvdi);

ym = y(1:453,1);         % 70% for modelling
m_t = y_t(1:453,1);
ym_org = y_org(1:453,1);
 
yv = y(454:584,1);       % 20% for validation
v_t = y_t(454:584,1);
vm_org = y_org(454:584,1); 

yt = y(585:end,1);        % 10% for test
t_t = y_t(585:end,1); 
yt_org = y_org(585:end,1); 

% Transformation of data 
ym_log = log(ym);
yv_log = log(yv);
yt_log = log(yt);
ym_yv = [ym; yv]; 

% Useful for prediction
ym_yv_t = [m_t; v_t];
ym_yv_log = log(ym_yv);
modelLim = length(ym)+1;        % Index for first data in validation set
testlim = length(ym_yv)+1;      % Index for first data in test set

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

%% (Simulate data for testing - the old versions) 
clc;
A_sim = input_arma.a;
C_sim = input_arma.c;
N = 10000; 
extraN = 100;
e = sqrt(1e-3) * randn(N+extraN,1);
sim_rain = filter(C_sim,A_sim, e);                          % For predicting input

KA = conv(model_B2.F,model_B2.D);
KB = conv(model_B2.D, model_B2.B);
KC = conv(model_B2.F,model_B2.C);
sim_nvdi = filter(KC, KA, e) + filter(KB, KA, sim_rain);    % For predicting output

sim_rain = sim_rain(extraN+1:end);
sim_nvdi = sim_nvdi(extraN+1:end);

%% 4.1 Recursive update of rain
% OLD
% Create Kalman for recursive estimation, based on previous prediction
% model for the rain. 
close all;
clc; 

% Data to put into Kalman
yx_input = xm_xv_log;    
% yx_input = sim_rain;                  % Change here fort testing simulated data
N = length(yx_input);      

% Prediction step and number unknownd
k = 1;                                  % Prediction step 
q0 = nnz(input_arma.a) - 1              % (3) - Number of unknowns in the A polynomial 
p0 = nnz(input_arma.c) - 1              % (4) - Number of unknowns in the C polynomial 

% Define the state space equations.
A = eye(p0+q0);                         % p0 + q0 are number of hidden states 
Rw = var(yx_input);                     % Try different! Measurement noise covariance matrix. Same dimension as Ry.
Re = 1e-6*eye(p0+q0);                   % Try different! System noise covariance matrix. 

% Set initial values
xt_t = [-0.1626 -0.2716 -0.5327 0.1274 0.3462 -0.2665 0.2634]';    
Rxt_t1 = 1*eye(p0+q0);                  % Initial covariance matrix, If large -> small trust initial values 

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
    Ct = [ -yx_input(t-1) -yx_input(t-2) -yx_input(t-36) ehat(t-1) ehat(t-2) ehat(t-6) ehat(t-9)];     % C_{t|t-1}
    
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
        Ck = [ -yx_t_input(t-1+k0) -yx_t_input(t-2+k0) -yx_t_input(t-36+k0) ehat(t+k0-1) ehat(t+k0-2) ehat(t+k0-6) ehat(t+k0-9)]; % C_{t+k|t}
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
%% 4.1 Recursive update of rain
% Examine the estimated parameters (Hidden states)
close all;
figure()
trueParams = [-0.1626 -0.2716 -0.5327 0.1276 0.3341 -0.2683 0.2605];     % For simulated data
% plotWithConf( (1:N-k), Xsave', xStd', trueParams);                     % For simulated data
plotWithConf( (1:N-k), Xsave', xStd');
legend('a1', 'a2', 'a36', 'c1', 'c2', 'c7', 'c9') 
title('Estimated states for input data rain')
xlim([37 N-k])

figure()
hold on 
plot(Xsave(:,50:end)')
xline(modelLim-50, 'r--', 'LineWidth', 1, 'Label','');
legend('a1', 'a2', 'a36', 'c1', 'c2', 'c7', 'c9')
xlim([1, length(Xsave)-50]);
hold off

fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(Xsave(:,1))
    fprintf('Estimated value: %5.2f (+/- %5.4f).\n', Xsave(k0,end), xStd(k0,end) )
end 

%% 4.1 Recursive update of rain
% FINAL recursive model for the rain 
% C1 PARAMETER HAS BEEN REMOVED!!

% Create Kalman for recursive estimation 
close all;
clc; 

% Data to put into Kalman
yx_input = xm_xv_log;    
% yx_input = sim_rain;                  % Change here fort testing simulated data
N = length(yx_input);      

% Prediction step and number unknownd
k = 7;                                  % Prediction step 
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
%% 4.1 Recursive update of rain
% Examine the estimated parameters (Hidden states)
close all;
figure()
% trueParams = [-0.1626 -0.2716 -0.5327 0.1276 0.3341 -0.2683 0.2605];   % For simulated data
% plotWithConf( (1:N-k), Xsave', xStd', trueParams);                     % For simulated data
plotWithConf( (1:N-k), Xsave', xStd');
legend('a1', 'a2', 'a36', 'c2', 'c7', 'c9') 
title('Estimated states for input data rain')
xlim([37 N-k])

figure()
hold on 
plot(Xsave(:,50:end)')
xline(modelLim-50, 'r--', 'LineWidth', 1, 'Label','');
legend('a1', 'a2', 'a36', 'c2', 'c7', 'c9')
xlim([1, length(Xsave)-50]);
hold off

fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(Xsave(:,1))
    fprintf('Estimated value: %5.2f (+/- %5.4f).\n', Xsave(k0,end), xStd(k0,end) )
end 
%% Plot k step prediction in "right" domain
close all; 

yx_input_org = exp(yx_input)-constant;
yxhatk_org = exp(yxhatk)-constant;

figure()
plot( [yx_input(modelLim:end) yxhatk(modelLim:end)] ) 
title( sprintf('%i-step prediction using the Kalman filter wrong domain', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 length(yx_input(modelLim:end))])

figure()
plot( [yx_input_org(modelLim:end) yxhatk_org(modelLim:end)] ) 
title( sprintf('%i-step prediction using the Kalman filter original domain', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 length(yx_input(modelLim:end))])
% axis([xv_t(1) xv_t(end) min(yx_input_org)*0.9 max(yx_input_org)*1.1])

%% Examine k-step prediction residual.
% it is not the best
error = yx_input_org(modelLim+20:end)-yxhatk_org(modelLim+20:end);  % Plotting hidden states above, seems to have reached a steady state after about 20 
plotACFnPACF( error, 40, sprintf('%i-step prediction using the Kalman filter', k)  );

fprintf('  The variance of original signal (still logged including model data) is %5.2f.\n', var(yx_input)')
fprintf('  The variance of original signal (still logged only validation) is %5.2f.\n', var(xv_log)')
fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(error)')

%% Kalman filter to recursively update parameters for nvdi BJ moddel
% OLD! 
% Initial guess of kalman filter using B2 model as guide 
close all;
clc;

KA_kalman = [-1.9126 1.1653 -0.2207 -0.1807 0.1982 -0.0489];
KB_kalman = [0.0223 0.0007 -0.0154 -0.0040 -0.0034];
KC_kalman = [-1.0970 0.2706]; 

% Data to put into Kalman
yx_input = xm_xv_log;
y_input = ym_yv_log;
% y_input = sim_nvdi;                   % Change here for testing simulated data
% yx_input = sim_rain;                  % Change here for testing simulated data
N = length(y_input); 

% Prediction step and number unknownd
k = k;                                  % OBS! Code is set up in way that k here MUST be same as above! (even though it we only need two less in prediction) 
nbr_params = length(KA_kalman) + length(KB_kalman) + length(KC_kalman);

% Define the state space equations.
A = eye(nbr_params);                    % Hidden states matrix 
Rw = var(y_input);                                % Try different! Measurement noise covariance matrix. Could use from noise estimate polynomial pred
Re = 1e-3*eye(nbr_params);              % Try different! System noise covariance matrix. 

% Set initial values
xt_t = [KA_kalman KB_kalman KC_kalman]';    
Rxt_t1 = 5*eye(nbr_params);             % Initial covariance matrix, IF large -> small trust initial values 

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
    Ct = [ -y_input(t-1) -y_input(t-2) -y_input(t-3) -y_input(t-36) -y_input(t-37) -y_input(t-38) yx_input(t-3) yx_input(t-4) yx_input(t-5) yx_input(t-39)  yx_input(t-40) ehat(t-1) ehat(t-2) ];
    
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
        Ck = [ -y_t_input(t+k0-1) -y_t_input(t+k0-2) -y_t_input(t+k0-3) -y_t_input(t+k0-36) -y_t_input(t+k0-37) -y_t_input(t+k0-38) yx_t_input2(t+k0-3) yx_t_input2(t+k0-4) yx_t_input2(t+k0-5) yx_t_input2(t+k0-39) yx_t_input2(t+k0-40) ehat(t+k0-1) ehat(t+k0-2)];
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


%% Examine the estimated parameters (for simulated data) 
close all;
figure()
% trueParams = [KA_kalman KB_kalman KC_kalman];     
% plotWithConf( (1:N-k), Xsave', xStd', trueParams);
plotWithConf( (1:N-k), Xsave', xStd');

title('Estimated states for output data nvdi')
xlim([42 N-k])

figure()
plot(Xsave(:,42:end)')
xline(modelLim-50, 'r--', 'LineWidth', 1, 'Label','');
legend('a1', 'a2', 'a3', 'a36', 'a37', 'a38', 'b3', 'b4', 'b5', 'b39', 'b40', 'c1', 'c2')
xlim([1, length(Xsave)-50]);

fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(Xsave(:,1))
    fprintf('Estimated value: %5.2f (+/- %5.4f).\n', xt(k0,end), xStd(k0,end) )
end 

%% Kalman filter to recursively update parameters for nvdi BJ moddel
% FINAL KALMAN
% REMOVED B4, B40, B5, B39, A1
close all;
clc;

KA_kalman = [-1.01 -0.0001 4.5 6.4 11.12];
KB_kalman = [4.23];
KC_kalman = [7.15 0.0001]; 

% Data to put into Kalman
yx_input = xm_xv_log;
y_input = ym_yv_log;
% y_input = sim_nvdi;                   % Change here for testing simulated data
% yx_input = sim_rain;                  % Change here for testing simulated data
N = length(y_input); 

% Prediction step and number unknownd
k = k;                                  % OBS! Code is set up in way that k here MUST be same as above! (even though it we only need two less in prediction) 
nbr_params = length(KA_kalman) + length(KB_kalman) + length(KC_kalman);

% Define the state space equations.
A = eye(nbr_params);                    % Hidden states matrix 
Rw = var(y_input);                      % Try different! Measurement noise covariance matrix. Could use from noise estimate polynomial pred
Re = 1*eye(nbr_params);                 % Try different! System noise covariance matrix. 

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
    Ct = [ -y_input(t-1) -y_input(t-3) -y_input(t-36) -y_input(t-37) -y_input(t-38) yx_input(t-3) ehat(t-1) ehat(t-2) ];
    
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
        Ck = [ -y_t_input(t+k0-1) -y_t_input(t+k0-3) -y_t_input(t+k0-36) -y_t_input(t+k0-37) -y_t_input(t+k0-38) yx_t_input2(t+k0-3) ehat(t+k0-1) ehat(t+k0-2)];
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

%% Examine the estimated parameters (for simulated data) 
close all;
figure()
% trueParams = [KA_kalman KB_kalman KC_kalman];     
% plotWithConf( (1:N-k), Xsave', xStd', trueParams);
plotWithConf( (1:N-k), Xsave', xStd');

title('Estimated states for output data nvdi')
xlim([42 N-k])

figure()
plot(Xsave(:,42:end)')
xline(modelLim-50, 'r--', 'LineWidth', 1, 'Label','');
legend('a1', 'a3', 'a36', 'a37', 'a38', 'b3', 'c1', 'c2')
xlim([1, length(Xsave)-50]);

fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(Xsave(:,1))
    fprintf('Estimated value: %5.2f (+/- %5.4f).\n', xt(k0,end), xStd(k0,end) )
end 

%% Plot k step prediction in "right" domain
close all; 

y_input_org = 1/2*(ym_yv+1)*(max_data - min_data)+min_data;
yhatk_org = exp(yhatk);
yhatk_org = 1/2*(yhatk_org+1)*(max_data - min_data)+min_data;

figure()
plot( [y_input(modelLim:end) yhatk(modelLim:end)] ) 
title( sprintf('%i-step prediction using the Kalman filter wrong domain', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 length(yx_input(modelLim:end))])

figure()
plot( [y_input_org(modelLim:end) yhatk_org(modelLim:end)] )
title( sprintf('%i-step prediction using the Kalman filter in original domain', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 length(yx_input(modelLim:end))])

%% Examine k-step prediction residual.
% it is not the best 
error = y_input_org(modelLim+40:end)-yhatk_org(modelLim+40:end);        % Steady state around 40 after 
mean_error = abs(mean(error))
plotACFnPACF( error, 40, sprintf('%i-step prediction using the Kalman filter', k)  );

fprintf('  The variance of original signal is %5.2f.\n', var(yx_input)')
fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(error)')


