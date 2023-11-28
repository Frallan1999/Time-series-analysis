%
% Time series analysis
% Computer exercise 2  
%
%
clear; 
close all;
% addpath('functions', '/data')     % Add this line to update the path
addpath('../functions', '../data')     % Add this line to update the path (Hanna)

%% 2.1 Modelling of exogenous input signal
%Generate some data following the Box-Jenkins model:

rng(0)
n = 500;                % Number of samples
A3=[1 .5];
C3 = [1 -.3 .2];
w= sqrt(2)*randn(n+100,1); 
x = filter (C3,A3,w);           % Create the input

A1 = [1 -.65];
A2 = [1 .90 .78];
C=1;
B=[0 0 0 0 .4];
e = sqrt(1.5) * randn(n + 100,1);
y = filter(C,A1,e) + filter(B,A2,x); % Create the output

x = x(101:end) , y = y(101:end) % Omit initial samples 
clear A1, A2, C, B, e, w, A3, C3

%% Plotting the data
close all
clc

m=50;
plot(x)
basicPlot(x,m,'x-data') 

%% Create an ARMA-model for the input xt as function of white noise
close all
clc

m=50;

%Initial guess 
%ACF ringing, PACF peak at 1 --> AR(1)?
na = 1;
nc = 0;
data = iddata(x);                   
model_x = armax(data, [na nc]); 
present(model_x);
e_hat = myFilter(model_x.a, model_x.c, x);
basicPlot(e_hat,m,'ARMA(1,0)')

whitenessTest(e_hat)

%Knowing it's an ARMA(1,2), test it with an ARMA(1,2) too 
%Model 1 according to instructions
na = 1;
nc = 2;
data = iddata(x);                   
x_arma = armax(data, [na nc]); 
present(x_arma);
e_hat = myFilter(x_arma.a, x_arma.c, x);
basicPlot(e_hat,m,'ARMA(1,2)')

whitenessTest(e_hat)

%We tested the same thing with idpoly and got the same result
% Seems that it's nice to use when wanting to control what orders are
% "active" in the polynomial

%% Pre-whitening
close all;
eps_t = myFilter(x_arma.a, x_arma.c, y); 
w_t = myFilter(x_arma.a, x_arma.c,x); 
basicPlot(eps_t,m,'Eps_t');
basicPlot(w_t,m,'W_t');

%% Compute CCF for eps_t = H(z) * w_t + v_t
close all;
M=40;
stem(-M:M,crosscorr(w_t ,eps_t,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

%% Determine suitable model orders for delay, A, B
% Delay: Delay exceeds confidence interval at lag 4, thus should be d = 4. 
% R (order for A2): Could be 2, as we have some ringing in the correlation.
% S: We see that it's decaying immedeatly, as s+d = time of decay, s = 0. 

A2 = [1 0 0]; 
B = [0 0 0 0 1];
Mi = idpoly ([1] ,[B] ,[] ,[] ,[A2]);
z = iddata(y,x);
Mba2 = pem(z,Mi); present(Mba2)
etilde = resid (Mba2, z );

%% Check etilde 
close all;
clc; 

%etilde and x should be uncorrelated --> Looks good, not sign diff from 0
M=40;
stem(-M:M,crosscorr(etilde.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% Is etilde white? 
% No, but we have now only modelled half of the BJ model, as y depends on
% something with x AND the error term.   
basicPlot(etilde.y,m,'etilde');
whitenessTest(etilde.y);

%% Plot etilde
close all;
clc; 

plot(etilde);
basicPlot(etilde.y,m,'etilde');

%% Determine orders for A1 and C1, model etilde = C1/A1 * e
%In the plot above, we suspect that we're dealing with an ARMA(1,0)
close all;
clc; 

A1 = [1 0]; 
C1 = [1];
model_init = idpoly (A1, [], C1);
etilde_data = iddata(etilde.y)
error_model = pem(etilde_data,model_init); 
present(error_model)
res_tilde = resid (error_model, etilde_data );
basicPlot(res_tilde.y,m,'ARMA(1,0)');
whitenessTest(res_tilde.y);

%% Reestimate the full model with pem
% Note that model will change when we do it all together again 

A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 0 1];
C = [1];
Mi = idpoly(1, B, C, A1, A2);
z = iddata(y,x);
MboxJ = pem(z,Mi);
present(MboxJ)
ehat = resid(MboxJ,z);      % the estimate of the noise process e_t

%% Final analysis of ehat
%ehat and x should be uncorrelated, looks OK (few outliers but ~95% conf)
M=40;
stem(-M:M,crosscorr(ehat.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

%% Final analysis of ehat contd.
%Whiteness of ehat -> super white!
whitenessTest(ehat.y);
basicPlot(ehat.y,m,'e-hat');

%To trust whiteness test, ehat needs to be normally distributed
%Test for normal distributed: data, label, type of test (D=default -->
%D'Augustino-Pearson K2 test, then level of alpha (1-alpha) = confidence)
checkIfNormal(ehat.y,'e-hat','D',0.05);

%% 2.2 Hairdryer data
clear; 
clc; 
close all; 
load tork.dat       % 1000 observations from an input-output experiment
% Data info:    
% input signal (voltage) - second column
% output signal (temperature) - first column 
% sampling distance 0.08 s.

% substract mean value, create object and plot first 300
tork = tork - repmat(mean(tork),length(tork),1);
y = tork(:,1); 
x = tork(:,2);
z = iddata(y,x);
n = length(x); 
plot(z(1:300))


%% Modelling the input x in Hairdryer 
clc;
close all; 

% First plotting the data to see if white and if not to know how to preceed
% in a potential ARMA modelling of the input
plot(x);
whitenessTest(x);
% --> No, so we need to form ARMA model of input

% Basic plots of input
m = 50; 
basicPlot(x,m,'Input data'); 

%% Modelling the input x in Hairdryer contd. 
clc;
close all;

%Initial guess 
%ACF ringing, PACF peak at 1 --> AR(1)?
na = 1;
nc = 0;
data = iddata(x);                   
model_x = armax(data, [na nc]); 
present(model_x);
e_hat = myFilter(model_x.a, model_x.c, x);
basicPlot(e_hat,m,'ARMA(1,0)')

whitenessTest(e_hat)

% --> Looks good to model this as an ARMA(1,0) (AR(1))

%% Prewhitening step (Multiplying with A3(z)/C3(z)
close all;
clc; 

eps_t = myFilter(model_x.a, model_x.c, y);      % eps_t = A3(z)/C3(z) * y
w_t = myFilter(model_x.a, model_x.c,x);         % wt = A3(z)/C3(z) * x
basicPlot(eps_t,m,'Eps_t');
basicPlot(w_t,m,'W_t');

%% Compute CCF for w_t to eps_t (eps_t = H(z) * w_t + v_t) 
close all;
clc; 

M=40;
stem(-M:M,crosscorr(w_t ,eps_t,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off

% From plot wee see that: d = 3, r = 1, s = 2 (5-3)
% As the CCF  yields an estimate of the impulse response, we can use this
% to determine suiteble orders for the delay, A2(z) and B(z). 
% Knowing the orders, we now want to estiamte the parameters given theese orders 
%% Determine suitable model orders for delay, A, B
% d (Delay): 3
% r (order for A2): Could be 1, slow decaying.
% s: decay starts at 5, as s+d = time of decay, s = 5-3 = 2. 

A2 = [1 0]; 
B = [0 0 0 1 0 0];
Mi = idpoly ([1] ,[B] ,[] ,[] ,[A2]);           % yt = B(z)z^-d/A2(z) * x_t (delay is in the B vector) 
Mi.Structure.b.Free = [zeros(1,4) 1 1];
z = iddata(y,x);
Mba2 = pem(z,Mi); 
present(Mba2)
etilde = resid(Mba2, z);

%% Check if models orders above are suitible 
% CCF between the input, xt, and the residual etilde_t (defined below) should be uncorrelated.
close all;
clc; 

%etilde and x should be uncorrelated 
M=40;
stem(-M:M,crosscorr(etilde.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off
% Looks good enough 

% Is etilde white? 
% No, but we have now only modelled half of the BJ model, as y depends on
% something with x AND the error term.   
basicPlot(etilde.y,m,'etilde');
whitenessTest(etilde.y);

%% Plotting e_tilde (Modelling the A1 and C1 polynomials) 
% We have now modeled yt as a function of the input xt, 
% but have not yet formed a model of the ARMA-process in the BJ model, i.e.,
% modeled the polynomials C1(z) and A1(z). We know e_tilde = C1/A1 e, so we
% want to find arma for e_tilde. As usual we start by studying the data

close all;
clc; 
m = 50; 

plot(etilde.y);
basicPlot(etilde.y,m,'etilde');

%% Modelling the A1 and C1 polynomials contd
close all;
clc; 

% Initial modell gues AR(1)
A1 = [1 0]; 
C1 = [1];
model_init = idpoly (A1, [], C1);
etilde_data = iddata(etilde.y)
error_model = pem(etilde_data,model_init); 
present(error_model)
res_tilde = resid (error_model, etilde_data );
basicPlot(res_tilde.y,m,'ARMA(1,0)');
whitenessTest(res_tilde.y);
% --> Looking good 

%% Reestimate the full model with pem
% We know have all our polynomials, and need to reestimate the parameters   

A1 = [1 0];
A2 = [1 0]; 
B = [0 0 0 1 0 0];
C = [1];                % Before named C1 
Mi = idpoly(1, B, C, A1, A2);
Mi.Structure.b.Free = [zeros(1,4) 1 1];
z = iddata(y,x);
MboxJ = pem(z,Mi);
present(MboxJ)
ehat = resid(MboxJ,z);      % the estimate of the noise process e_t

%% Final analysis of ehat (is our entier model good enough?)
%ehat and x should be uncorrelated, looks OK (few outliers but ~95% conf)
M=40;
stem(-M:M,crosscorr(ehat.y ,x ,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n)*ones(1,2*M+1), 'r--') 
plot(-M:M, -2/sqrt(n)*ones(1,2*M+1),'r--') 
hold off


%% Final analysis of ehat contd.
%Whiteness of ehat -> white, but not normal, can we trust? 
whitenessTest(ehat.y);
basicPlot(ehat.y,m,'e-hat');

%To trust whiteness test, ehat needs to be normally distributed
%Test for normal distributed: data, label, type of test (D=default -->
%D'Augustino-Pearson K2 test, then level of alpha (1-alpha) = confidence)
checkIfNormal(ehat.y,'e-hat','D',0.05);

%% 2.3 Prediction of ARMA-processes
close all; 
clear; 
clc; 
load svedala; 
y = svedala; 

% data sampled every hour, with estimated mean value subtracted (11.35◦ C) 

% Suitable model parameters for the data set are
A = [ 1 -1.79 0.84 ]; 
C = [ 1 -0.18 -0.11 ];

%% One step prediction 
k = 1;      % prediction step 

[Fk, Gk] = polydiv( C, A, k );  % solves the Diophantine equation (different for each k) 
yhat_1 = filter( Gk, C, y );    % k-step prediction, y_hat_t+k|t, 
yhat_1 = yhat_1(4:end)          % Is it 4 because length(c) + k? 
var_1 = var(y(4:end) - yhat_1)


%% k-step prediction using k = 3 
k3 = 3;      % predicti

m = 50; 

[Fk3, Gk3] = polydiv( C, A, k3 );       % solves the Diophantine equation (different for each k) 
yhat_3 = filter( Gk3, C, y );           % k-step prediction, y_hat_t+k|t, 
yhat_3 = yhat_3(6:end);                 % Is it 4 because length(a) + k? 
error_3 = y(6:end) - yhat_3; 

mean3 = mean(error_3)                           
variance3 = var(error_3)
theoretical_variance3 = sum(Fk3.^2) * var_1
conf_3 = 2*sqrt(theoretical_variance3);
conf_int3 = [mean3-conf_3, mean3+conf_3]

error3_outside = (sum(error_3>conf_int3(2)) + sum(error_3<conf_int3(1)))/length(error_3)  

% 1. Estimated mean and the expectation of the prediction error
% 2. Comment on difference in variance: somewhat higher theoretical
% 3. Determine the theoretical 95% confidence interval
% 4. How large percentage of the prediction errors are outside the 95%

% 5. plotting
figure(1)
plot(y);
hold on; 
plot(yhat_3);
title('process and 3 step prediction')
hold off;

figure(2)
plot(error_3)
figure(3)
basicPlot(error_3, m, '3 step error')
% We see an MA(2) 

%% k-step prediction using k = 26
k26 = 26;      % prediction step 
close all; 

[Fk26, Gk26] = polydiv( C, A, k26 );    % solves the Diophantine equation (different for each k) 
yhat_26 = filter( Gk26, C, y );            % k-step prediction, y_hat_t+k|t, 
yhat_26 = yhat_26(29:end);                  % Is it 4 because length(a) + k? 
error_26 = y(29:end) - yhat_26;

mean26 =  mean(error_26)
variance26 = var(error_26)
theoretical_variance26 = sum(Fk26.^2) * var_1
conf_26 = 2*sqrt(theoretical_variance26);
conf_int26 = [mean26-conf_26, mean26+conf_26]

error26_outside = (sum(error_26>conf_int26(2)) + sum(error_26<conf_int26(1)))/length(error_26)

% 1. Estimated mean and the expectation of the prediction error
% 2. Comment on difference in variance: much higher theoretical
% 3. Determine the theoretical 95% confidence interval
% 4. How large percentage of the prediction errors are outside the 95%

% 5. plotting
m = 100;
figure(1)
plot(y);
hold on; 
plot(yhat_26);
title('process and 26 step prediction')
hold off;

figure(2)
plot(error_26)
figure(3)
basicPlot(error_26, m, '26 step error')
% We see a periocidity of 25

%% 2.4 Prediction of ARMAX-processes
close all;
clear; 
clc;

load svedala
load sturup    % 3 step prediction, can be used as external signal to svedala
x = sturup;
y = svedala; 

% Set model parameters 
A= [ 1 -1.49 0.57 ];
B = [ 0 0 0 0.28 -0.26 ]; 
C= [1];

basicPlot(sturup,50, 'sturup')
% How large is the delay in this temperature model? 3 - how do we know? 

%% Form the k - step prediction 
% using k = 3 and 26
close all; 
clc; 

k = 3;
% k = 26; 

[Fk, Gk] = polydiv( C, A, k );  % solves the (old) Diophantine equation (different for each k) 
[Fk_hat, Gk_hat] = polydiv(conv(B,Fk), C, k);  % solves the (old) Diophantine equation (different for each k) 

yhat = filter(Fk_hat, [1], x) + filter(Gk_hat, C, x) + filter(Gk, C, y);   
% In second term above, why do we input x when it is a 4 step prediction?? 

yhat = yhat(9+k:end);       % 9 is order of Gk_hat as that is order of C 
% How do we know what order to remove??

error = y(9+k:end) - yhat; 
var(error)

% 5. plotting
figure(1)
plot(y);
hold on; 
plot(yhat);
title('process and 3 step prediction')
hold off;

m = 50; 
figure(2)
plot(error)
basicPlot(error, m, '3 step prediction error')
% We see an MA(2) 

%% If we forget one term in y prec´diction
close all; 

yhat_wrong =  filter(Gk_hat, C, x) + filter(Gk, C, y);   
yhat_wrong = yhat_wrong(9+k:end);       % 9 is order of Gk_hat as that is order of C 

figure(1)
plot(yhat_wrong);
hold on; 
plot(yhat);
hold off;

% For k = 3, fk_hat is zero and we can see no difference, 
% For k = 26, we see that we get a super small amplitude for the wrong one 

%% 2.5 Prediction of SARIMA-processes
close all;
clear; 
clc;

load svedala
y = svedala; 

m = 100;
plot(y)
basicPlot(y,m,'svedala')
% We see a season of 24 hours 

%% Model svedala with differentiation
A24 = [ 1 zeros(1, 23) -1 ];

y_s = filter(A24, 1, y);            % filter on seasonality 12
y_s = y_s(length(A24):end);
y_data = iddata(y_s);
figure()
plot(y_s);
basicPlot(y_s, 50, "y_s");          % Basic analysis through plots 

% From comp ex 1. 
model_init = idpoly([1 0 0],[],[1 zeros(1,23) 1]);
model_init.Structure.c.Free = [zeros(1,24) 1];
found_model = pem(y_data,model_init)
res = resid(found_model, y_data);             % Create new residual
basicPlot(res.y, m, "residual"); 
present(found_model);
whitenessTest(res.y);
checkIfNormal(res.y,"Final model");

%% Create predictions from found model
close all; 
clc;

% Parameters from our found model 
A = found_model.a;
C = found_model.c;

% Prediction
k = 26; 
[Fk, Gk] = polydiv( C, A, k );  % solves the Diophantine equation (different for each k) 
yhat_k = filter( Gk, C, y_s );    % k-step prediction, y_hat_t+k|t, 
yhat_k = yhat_k(length(C)+k:end)          % Is it length(c) + k? 
var_k = var(y_s(length(C)+k:end) - yhat_k)

% Better for k = 26 but not for k = 3. 

