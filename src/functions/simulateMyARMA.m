% Filters noice to simulate arma process 

function y = simulateMyARMA(C, A, sigma2, N)

rng(0);
e = sqrt(sigma2) * randn(N,1);  
y = filter(C, A, e); 
y = y(101:end);

end
