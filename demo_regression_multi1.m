clear all;
close all;

M = 5;  % number of annotators
N = 100; % maximum number of samples
O = 15; % number of samples per annotator
noise = log(linspace(0.1,2,M)');
%noise = log(0.5^2*ones(M,1));
epsilon = 0.1;

% create data
x = linspace(0,1,N)';
%y = spalloc(N,M,M*O);
y = NaN(N,M);
for m=1:M
    r = randperm(length(x)); r = r(1:O);    % select O number of random indices
        
    % Method 1: add input independent noise to function evaluation
    y(r,m) = onevar(x(r,:)) + normrnd(0,exp(noise(m)),O,1);       
end
unusedinds = sum(isnan(y),2)==M;
x(unusedinds,:) = [];
y(unusedinds,:) = [];

xt = linspace(0,1,100)';


% ---------------------------
% --- Construct the model ---
% 
% First create structures for Gaussian likelihood and squared
% exponential covariance function with ARD
lik = lik_mgaussian('ndata',M,'sigma2', 0.2^2*ones(1,M));
gpcf = gpcf_sexp('lengthScale', 0.15, 'magnSigma2', 23);
pl = prior_unif();
pm = prior_sqrtunif();
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Following lines do the same since the default type is FULL
mgp = gp_set('type','FULL','lik',lik,'cf',gpcf);



% --- MAP estimate using scaled conjugate gradient algorithm ---
disp(' MAP estimate for the parameters')
% Set the options for the scaled conjugate optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
% Optimize with the scaled conjugate gradient method
mgp=gp_optim(mgp,x,y,'optimf',@fminscg,'opt',opt);


% average data
nanindy = isnan(y);
y0 = y; y0(nanindy)=0;
yavg = sum(y0,2)./sum(~nanindy,2);

% ---------------------------
% --- Construct the model ---
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', [0.15], 'magnSigma2', 0.2^2);
pl = prior_unif();
pm = prior_sqrtunif();
gpcf = gpcf_sexp(gpcf, 'lengthScale_prior', pl, 'magnSigma2_prior', pm);
% Following lines do the same since the default type is FULL
gp_avg = gp_set('type','FULL','lik',lik,'cf',gpcf);
gp_avg=gp_optim(gp_avg,x,yavg,'optimf',@fminscg,'opt',opt);


symbols = ['x','o','+','*','s','d','v','^','<','>'];
colors = ['b','g','r','c','m','k','y'];
figure; hold on;
for m=1:M
  im = ~isnan(y(:,m));
  xm = x(im); 
  ym = y(im,m);
  plot(xm,ym,[symbols(m),colors(m)]);
end
[mu,s2] = gp_pred(mgp, x, y, xt);
gpPlot(xt,mu,s2);
plot(xt, onevar(xt),'k');
legend('Location','Northwest')
title('Multi-annotator GP model');


figure; hold on;
for m=1:M
  im = ~isnan(y(:,m));
  xm = x(im); 
  ym = y(im,m);
  plot(xm,ym,[symbols(m),colors(m)]);
end
[mu,s2] = gp_pred(gp_avg, x, yavg, xt);
gpPlot(xt,mu,s2);
plot(xt, onevar(xt),'k');
legend('Location','Northwest')
title('Average GP model');