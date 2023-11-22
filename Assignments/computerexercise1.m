%
% Time series analysis
% Computer exercise 1  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%%  2.1 Working with time series in Matlab
% create A and C polynomials for ARMA process 
A1 = [1 -1.79 0.84];
C1 = [1 -0.18 -0.11];
A2 = [1 -1.79];
C2 = [1 -0.18 -0.11];

arma_1 = idpoly(A1, [], C1);        % Create arma polynomial 1
arma_2 = idpoly(A2, [], C2);        % Create arma polynomial 2

arma_1.a            % way of fetching a polynomial
figure(1)
pzmap(arma_1)       % view poles and zeros of arma

% Simulation of ARMA texst
rng(0);             % set seed
sigma2 = 1;         % variance of error terms
N = 100;            % length of simulated samples
e = sqrt(sigma2) * randn(N,1);       % generate normal distributied noice, 
y = filter(arma_1.c, arma_1.a, e);   % simulating an ARMA process

% NOTE: always simulate longer process than needed when simulating a process
% containing an AR part, and then omitt the initial samples. Prefer ab 
% exaggerated number of omitted samples. Assume inital effects will be
% negligible after say 100 samples 

% Now we simulate using our created function "simulateMyARMA" 
y_hat = simulateMyARMA(arma_1.c, arma_1.a, sigma2, N);

% Now we want to simulate what we will be using going forward
N = 300; 
sigma2 = 1.5; 
y1 = simulateMyARMA(arma_1.c, arma_1.a, sigma2, N);
y2 = simulateMyARMA(arma_2.c, arma_2.a, sigma2, N);

%% Question 1
close all

figure(1)
subplot(211)
plot(y1)
subplot(212)
plot(y2)

figure(2)
subplot(211)
pzmap(arma_1)
subplot(212)
pzmap(arma_2)

% we can see that y2 process diverges. When studying the poles and zeros for
% that arma, we see a pole outside the unit circle. 

%% Question 2
close all
% Information about covariance.
% Theoretical: The "kovarians" function in matlab can be used to calculate 
% the theoretical covariance function r_y(k) for an arma process. Function 
% assumes that the driving noise process has unit variance, i.e. V(et) =
% sigma2 = 1. 
% Estimated: use the function r_est = covf(y,m)

% Finding theoretical and estimated covariance for arma_1
m = 20;         %m is the maximum lag value
r_theo = kovarians(arma_1.c, arma_1.a, m);      % caluclate theoretical covariance function
stem(0:m, r_theo*sigma2);
hold on
r_est = covf( y1, m+1 )         % calculate estimated covariance function
stem(0:m, r_est, 'r');

% Question: Why are the estimated and theoretical covariance functions not
% identical?  
% Answer: .... 

%Remember that for the estimated ACF we should only use lags up to N/4. 
              
%% Question 3 - plot ACF PACF and normplot
close all
% call on function that does basic analysis by plotting the acf, pacf, 
% and normplot of your data. 
basicPlot(y1,m,'y1')

%% Question 3 - estimate variables given plot above
close all
na = 2;
nc = 0;     % For now
data = iddata(y1);                  % make data an object type for estimation (not needed here)
ar3_model = arx(y1, na);             % estimate model using LS method arx for AR(na) 
arma_model = armax(y1, [na nc]);    % estimate model using MS method for ARMA(na, ca)

present(ar3_model)       % display estimated parameters, their std and model FPE
present(arma_model)     % display estimated parameters, their std and model FPE

% Calculate error residual of estimated model, note that we switched places
% of a and c polynomials to get the inverse
e_hat = filter (arma_model.a, arma_model.c, y);  %We wonder what to send in here - should it be poly and just y? 

%How do we see that they are corrupted up to na?
basicPlot(e_hat(1:20), m, 'Residuals')
figure()
plot(e_hat(1:20))

%Remove the na first error estimations. We create a separate model for
%this for future use -> myFilter. 
e_hat = e_hat(length(arma_model.a):end);

%% Question 3 contd. testing different models
close all 

% Testing AR(2) (Best) 
na = 2;
ar3_model = arx(y1, na);  
e_hat_ar = myFilter(ar3_model.a, ar3_model.c, y1);
basicPlot(e_hat_ar,m,'AR(2)');
present(ar3_model)
% Looks good, FPE: 1.587 

% Testing ARMA(2,1) - see a bit of ringing in both ACF and PACF
na = 2;
nc = 1; 
arma_model = armax( y1, [na nc]);    
e_hat_arma = myFilter(arma_model.a, arma_model.c, y1);
basicPlot(e_hat_arma, m, 'ARMA(2,1)')
present(arma_model)
% Worse FPE here than in the AR models, 1.595, c1 parameter almost
% unsignificant

%% 2.2 Model order estimation of an ARMA-process
clear
close all
clc
load data.dat           % 200 observations from of arma(1,1)-process
load noise.dat          % noise for generating the data
data = iddata(data);    % make data an object
m = 20;                 % nbr of lags

%% Question 4
% try to fit for AR(p) p = 1...5 
for p=1:5 
    ar_model = arx(data, p);            % estimate AR model of order p 
    ar_model = ar_model(p:end);         % remove n inital samples (as we now do this in filter)
    rar = resid(ar_model, data);        % directly computes the residual for the given model 

    %Plots the residuals and the noise together as well as parameter
    %estimates
    figure(); 
    plot(rar);
    hold on; 
    plot(noise, 'r');
    title(p);
    basicPlot(rar.y, m, p + "residual")
    present(ar_model)
end

% try modelling data as AR(p) model. What does this mean we should do?
% FPE and significance for the different ps 
% p = 1, FPE = 1.782, all significant
% p = 2, FPE = 1.388, all significant
% p = 3, FPE = 1.282, all significant <-- chosen, not too different from 4
% but fewer parameters
% p = 4, FPE = 1.239, all significant
% p = 5, FPE = 1.245, 5th not significant

%% Question 5
close all
%  model the data using an ARMA(p, q)–models, for p, q = 1, 2,
p = 1; 
q = 2;

arma11_model = armax(data, [p q]);
arma11_model = arma11_model(p:end);
present(arma11_model)
% FPE: 1.182, all signifiacnt

% Which model would we use? (not knowing it is an ARMA)

ar3_model = arx(data, 3);            % estimate AR model of order p 
ar3_model = ar3_model(3:end);         % remove n inital samples (as we now do this in filter)
figure(1)
res_ar = resid(ar3_model, data)
figure(2)
res_arma = resid(arma11_model, data)

%% Estimation of a SARIMA-process
clear
close all
clc

rng(0)                          % Sets the random seed to 0
A = [1 -1.5 0.7];               % Sets A polynomial
C = [1 zeros(1,11) -0.5];       % Sets C polynomial
A12 = [1 zeros(1,11) -1];       % Sets the season
A_star = conv(A,A12);
e = randn(10000 ,1);              % Create noise
y = filter(C,A_star,e);         % Generate the process
y = y(101:end);                 
plot(y)

m = 20;                         % Number of lags
basicPlot(y, m, "SARIMA");      % Basic analysis through plots     
% we see it is ringing in acf and pacf, indicating seasonality. 
% to model the process taking the season into account ->
% create a differentiated process. remove season first, and then create
% object

%% Removing seasonality in data
close all
y_s = filter(A12,1,y);          % Filter on seasonality 12 
y_s = y_s(length(A12):end);     % Omit initial samples
data = iddata(y_s);             % Create object for estimation
figure(1)
plot(y_s)
basicPlot(y_s, m, "no season");      % Basic analysis through plots     

%% Create model
close all
model_init = idpoly([1 0 0] ,[] ,[]);       % Set up inital model
model_armax = pem(data,model_init);         % Estimate a1 and a2 
res = resid(model_armax, data);             % Create residual
basicPlot(res.y, m, "residual");            % NOTE: we still se something at lag 12

% now estimate a1,a2 AND c12
model_init = idpoly([1 0 0],[],[1 zeros(1,12)]);
model_init.Structure.c.Free = [zeros(1,12) 1];
model_armax = pem(data,model_init)

res = resid(model_armax, data);             % Create new residual
basicPlot(res.y, m, "residual"); 
present(model_armax);
figure(3)
whitenessTest(res.y);
% we now see that the seasonality in 12 is removed. 
% The residual looks white enough 
% Are the ACF and/or PACF coefficients Gaussian distributed so that you can 
% trust your whiteness test? Are the estimated parameters significant?

%% Typical modelling steps
% 1. Is there a trend? Try removing it.
% 2. Is there any seasonality? Try removing it.
% 3. Iterate between
% (a) Which is the lowest order strong AR- or MA-component? Try removing it 
% by including it in the model. Always begin with the strongest AR-component, 
% then inspect the MA-components in the next iteration.
% (b) Is the residual white noise? If not, go to (a). Can you trust your test,
% i.e., is the ACF and/or PACF Gaussian distributed? 
% If not, what are the consequences?
% 4. Are all parameter estimates statistically significant? If not, redo 
% the analysis and use a smaller model (order).

%% 2.3 Estimation of real data
clear
close all
clc
load svedala

data = svedala;
%% 1. Is there a trend? 
figure(1)
plot(data)
% seems so yes, lets estimate it and try o remove it from the data
N = length(svedala);
X = [ ones(N,1) (1:N)' ];
theV = inv( X'*X )*X'*data;             % This is the least-squares estimate of the trend. 
z = data - theV(1) - theV(2)*(1:N)';    % Subtract the estimated trend.

% Add the estimated trend to the plot.
figure(1)
hold on
plot(theV(1) + theV(2)*(1:N)', 'r')
legend('Data', 'Linear trend', 'Location', 'SE')
hold off

% Plot the resulting de-trended data and its ACF and PACF.
figure(2);
plot(z);

%% 2. Is there any seasonality? 
close all
m = 200;     % number of lags (upp this to find seasonality)
plot(z)
basicPlot(data,m, "before removal of trend")
basicPlot(z, m, "after removal of trend")
% See a seasonality of 24, either we remove this or we incorporate this in
% our model. 

% Start by removing sesason
A24 = [1 zeros(1,23) -1];
z_s = filter(A24, 1, z);
z_s = z_s(length(A24):end);
% z_s = iddata(z_s);
figure()
plot(z_s);
basicPlot(z_s, 30, "z_s");      % Basic analysis through plots     

%% 3. Iterate between (a) and (b)
close all
m = 30;

% initial model, estimate a1 and a2
model_init = idpoly([1 0 0] ,[] ,[]);       % Set up inital model
model_armax = pem(data,model_init);         % Estimate a1 and a2 
res = resid(model_armax, data);             % Create residual
basicPlot(res.y, m, "residual");            % Does not seem  wite
present(model_armax);

% now estimate a1,a2 AND c24
model_init = idpoly([1 0 0],[],[1 zeros(1,23) 0]);
model_init.Structure.c.Free = [zeros(1,23) 1 1];
model_armax = pem(data,model_init)

res = resid(model_armax, data);             % Create new residual
basicPlot(res.y, m, "residual"); 
present(model_armax);
whitenessTest(res.y);