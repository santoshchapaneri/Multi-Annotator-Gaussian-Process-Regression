function gpPlot(X,mu,s2)

if isempty(X)
  X = linspace(1,length(mu),length(mu))';  
end

hold on;
fill([X;flipdim(X,1)], [mu+2*sqrt(s2);flipdim([mu-2*sqrt(s2)],1)],[0.9 0.9 0.9]); 
plot(X,mu,'k--');
