function basicPlot(data, noLags, titleStr, signLvl)

% If the fourth argument is not given, set the value to its default.
if nargin<4
    signLvl = 0.05;
end

figure
subplot(311)
acfEst = acf( data, noLags, signLvl, 1 );
title( sprintf('ACF (%s)',titleStr))
subplot(312)
pacfEst = pacf( data, noLags, signLvl, 1 );
title( sprintf('PACF (%s)',titleStr))
subplot(313)
normplot(data)
title(sprintf('Normplot (%s)',titleStr))

end