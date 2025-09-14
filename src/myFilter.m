%This function does filtering and omitts the na first samples of the output
%process. 

%NOTE that we don't want to use this in some cases, especially in the case of prediction.
%

function e_hat = myFilter(poly_a, poly_c, data)

e_hat = filter(poly_a, poly_c, data);

e_hat = e_hat(length(poly_a):end);

end