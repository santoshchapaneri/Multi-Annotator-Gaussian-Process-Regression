function lik = lik_mgaussian(varargin)
%LIK_MGAUSSIAN  Create a likelihood structure combining multiple Gaussian
%likelihoods
%
%  Description
%    LIK = LIK_MGAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a multi-Gaussian likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values. Obligatory
%    parameter is 'ndata', which tells the number of components.
%
%    LIK = LIK_MGAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihood function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for multi-Gaussian likelihood function [default]
%      sigma2       - variance [1 x ndata vector of 0.1s]
%      sigma2_prior - prior for sigma2 [1 x ndata cell of prior_logunif]
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, PRIOR_*, LIK_*

% Internal note: Because Gaussian noise can be combined
% analytically to the covariance matrix, lik_mgaussian is internally
% little between lik_* and gpcf_* functions.
  
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_MGAUSSIAN';
  ip.addOptional('lik', [], @isstruct);
  ip.addParamValue('ndata',[], @(x) isscalar(x) && x>0 && mod(x,1)==0);
  ip.addParamValue('sigma2',[], @(x) isvector(x) && all(x>0));
  %ip.addParamValue('sigma2_prior',[], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('sigma2_prior',{}, @(x) ((iscell(x) && ~isempty(x)>0) || isequal(x,[]) || isstruct(x)));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'multi-Gaussian';
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'multi-Gaussian')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('ndata',ip.UsingDefaults)
    ndata = ip.Results.ndata;
    lik.ndata=ndata;    
  end
  if isempty(lik.ndata)
    error('NDATA has to be defined')
  end  
  if init || ~ismember('sigma2',ip.UsingDefaults)
    sigma2=ip.Results.sigma2;
    if isempty(sigma2) 
      lik.sigma2 = 0.1*ones(ndata,1);
    else
      if (size(sigma2,2) == lik.ndata && size(sigma2,1) == 1)
        lik.sigma2 = sigma2;
      else
        error('The size of sigma2 has to be 1xNDATA')
      end
    end    
  end

  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    sigma2_prior=ip.Results.sigma2_prior;  
    if isequal(sigma2_prior,{}) 
      lik.p.sigma2=repmat({prior_logunif()},1,lik.ndata);
    else
      if (isstruct(sigma2_prior) || isequal(sigma2_prior,[]))
        lik.p.sigma2=repmat({sigma2_prior},1,lik.ndata);
      elseif (size(sigma2_prior,2) == lik.ndata && size(sigma2_prior,1) == 1)
        lik.p.sigma2=ip.Results.sigma2_prior;
      else
        error('The size of sigma2_prior has to be 1xNDATA')
      end  
    end
  end
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_mgaussian_pak;
    lik.fh.unpak = @lik_mgaussian_unpak;
    lik.fh.lp = @lik_mgaussian_lp;
    lik.fh.lpg = @lik_mgaussian_lpg;
    lik.fh.cfg = @lik_mgaussian_cfg;
    lik.fh.trcov  = @lik_mgaussian_trcov;
    lik.fh.trvar  = @lik_mgaussian_trvar;
    lik.fh.recappend = @lik_mgaussian_recappend;
    lik.fh.y = @lik_mgaussian_y;
    lik.fh.yg = @lik_mgaussian_yg;
    lik.fh.e = @lik_mgaussian_e;
    lik.fh.eg = @lik_mgaussian_eg;
  end

end

function [w s] = lik_mgaussian_pak(lik)
%lik_mgaussian_PAK  Combine likelihood parameters into one vector.
%
%  Description
%    W = lik_mgaussian_PAK(LIK) takes a likelihood structure LIK
%    and combines the parameters into a single row vector W.
%
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%     
%  See also
%    lik_mgaussian_UNPAK

  w = []; s = {};
  wh = []; sh = {};
%   for i=1:length(lik.sigma2)
%     if ~isempty(lik.p.sigma2{i})    
%       w = [w log(lik.sigma2(i))];
%       s = [s; 'log(gaussian.sigma2)'];
%       % Hyperparameters of sigma2
%       [wh sh] = lik.p.sigma2{i}.fh.pak(lik.p.sigma2{i});
%       w = [w wh];
%       s = [s; sh];
%     end
%   end    

  % likelihood parameters sigma2
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})    
      w = [w log(lik.sigma2(i))];
      s = [s; 'log(gaussian.sigma2)'];
    end
  end
  % Hyperparameters of sigma2
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})                
      [wh sh] = lik.p.sigma2{i}.fh.pak(lik.p.sigma2{i});
      w = [w, wh];
      s = [s; sh];
    end    
  end  
end

function [lik, w] = lik_mgaussian_unpak(lik, w)
%lik_mgaussian_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = lik_mgaussian_UNPAK(LIK, W) takes a likelihood structure
%    LIK and extracts the parameters from the vector W to the LIK
%    structure.
%
%    Assignment is inverse of  
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]' x ndata
%
%  See also
%    lik_mgaussian_PAK
  
  % likelihood parameters sigma2
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})    
      lik.sigma2(i) = exp(w(1));
      w = w(2:end);
    end
  end
  % Hyperparameters of sigma2
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})          
      [p, w] = lik.p.sigma2{i}.fh.unpak(lik.p.sigma2{i}, w);
      lik.p.sigma2{i} = p;
    end
  end
end

function lp = lik_mgaussian_lp(lik)
%lik_mgaussian_LP  Evaluate the log prior of likelihood parameters
%
%  Description
%    LP = LIK_MGAUSSIAN_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters.
%
%  See also
%    lik_mgaussian_PAK, lik_mgaussian_UNPAK, lik_mgaussian_G, GP_E

  lp = 0;
  
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})      
      lp = lp + log(lik.sigma2(i)) + ...
           lik.p.sigma2{i}.fh.lp(lik.sigma2(i), lik.p.sigma2{i}) ;
    end
  end
end

function lpg = lik_mgaussian_lpg(lik)
%lik_mgaussian_LPG  Evaluate gradient of the log prior with respect
%                  to the parameters.
%
%  Description
%    LPG = lik_mgaussian_LPG(LIK) takes a Gaussian likelihood
%    function structure LIK and returns LPG = d log (p(th))/dth,
%    where th is the vector of parameters.
%
%  See also
%    lik_mgaussian_PAK, lik_mgaussian_UNPAK, lik_mgaussian_E, GP_G

  lpg = [];
  lpg_prior = [];

  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})        
      lpgs = lik.p.sigma2{i}.fh.lpg(lik.sigma2(i), lik.p.sigma2{i});
      lpg = [lpg lpgs(1).*lik.sigma2(i) + 1];
      if length(lpgs) > 1
        lpg_prior = [lpg_prior lpgs(2:end)];
      end            
    end
  end
  lpg = [lpg lpg_prior];
end

function DKff = lik_mgaussian_cfg(lik, x, x2, y)
%lik_mgaussian_CFG  Evaluate gradient of covariance with respect to
%                 Gaussian noise
%
%  Description
%    Gaussian likelihood is a special case since it can be
%    analytically combined with covariance functions and thus we
%    compute gradient of covariance instead of gradient of likelihood.
%
%    DKff = lik_mgaussian_CFG(LIK, X) takes a Gaussian likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of Gaussian noise covariance
%    matrix Kff = k(X,X) with respect to th (cell array with
%    matrix elements).
%
%    DKff = lik_mgaussian_CFG(LIK, X, X2) takes a Gaussian
%    likelihood function structure LIK, a matrix X of input
%    vectors and returns DKff, the gradients of Gaussian noise
%    covariance matrix Kff = k(X,X) with respect to th (cell
%    array with matrix elements).
%
%  See also
%    lik_mgaussian_PAK, lik_mgaussian_UNPAK, lik_mgaussian_E, GP_G

  [n, m] = size(x);  
  sn2hat = 1./sum(bsxfun(@rdivide,~isnan(y),lik.sigma2),2);  
  
  DKff = {};
  j=0;
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})
      j=j+1;        
      dsn2hat = (sn2hat.^2)./lik.sigma2(i); 
      dsn2hat(isnan(y(:,i)))=0;
      DKff{j} = sparse(1:n,1:n, dsn2hat,n,n);
    end
  end
end

function DKff  = lik_mgaussian_ginput(lik, x, t, g_ind, gdata_ind, gprior_ind, varargin)
%lik_mgaussian_GINPUT  Evaluate gradient of likelihood function with 
%                     respect to x.
%
%  Description
%    DKff = lik_mgaussian_GINPUT(LIK, X) takes a likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of likelihood matrix Kff =
%    k(X,X) with respect to X (cell array with matrix elements)
%
%    DKff = lik_mgaussian_GINPUT(LIK, X, X2) takes a likelihood
%    function structure LIK, a matrix X of input vectors and
%    returns DKff, the gradients of likelihood matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%
%  See also
%    lik_mgaussian_PAK, lik_mgaussian_UNPAK, lik_mgaussian_E, GP_G

end

function C = lik_mgaussian_trcov(lik, x, y)
%lik_mgaussian_TRCOV  Evaluate training covariance matrix
%                    corresponding to Gaussian noise
%
%  Description
%    C = lik_mgaussian_TRCOV(LIK, TX, Y) takes in likelihood function
%    of a Gaussian process GP and matrices TX, Y that contains
%    training input and output vectors respectively. Returns covariance 
%    matrix C. Every element ij of C contains covariance between inputs 
%    i and j in TX
%
%  See also
%    lik_mgaussian_COV, lik_mgaussian_TRVAR, GP_COV, GP_TRCOV

  [n, m] = size(x);  
  s2hat = 1./sum(bsxfun(@rdivide,~isnan(y),lik.sigma2),2);
  C = sparse(1:n,1:n, s2hat,n,n);
end

function C = lik_mgaussian_trvar(lik, x, y)
%lik_mgaussian_TRVAR  Evaluate training variance vector
%                    corresponding to Gaussian noise
%
%  Description
%    C = lik_mgaussian_TRVAR(LIK, TX) takes in covariance function
%    of a Gaussian process LIK and matrix TX that contains
%    training inputs. Returns variance vector C. Every element i
%    of C contains variance of input i in TX
%
%
%  See also
%    lik_mgaussian_COV, GP_COV, GP_TRCOV
  
  C = 1./sum(bsxfun(@rdivide,~isnan(y),lik.sigma2'),2);
end

function C = lik_mgaussian_y(lik, x, y)
%lik_mgaussian_Y  Converts multi-annotated y into single reweighted
%                 output vector
%
%  Description
%    C = lik_mgaussian_y(LIK, X, Y) takes in likelihood function
%    and input and multi-annotated output training data X, Y. Returns 
%    reweighted output vector.

  s2hat = 1./sum(bsxfun(@rdivide,~isnan(y),lik.sigma2),2);
  y(isnan(y))=0;
  C = s2hat.*sum(bsxfun(@rdivide,y,lik.sigma2),2);
end

function g = lik_mgaussian_yg(lik, x, y)
%lik_mgaussian_yg  Gradient of converted multi-annotated y into single 
%                  reweighted output vector with respect to the Gaussian 
%                  noise parameters
% 
%  Description
%    C = lik_mgaussian_yg(LIK, X, Y) takes in likelihood function
%    and input and multi-annotated output training data X, Y. Returns 
%    reweighted output vector.

  indnany = isnan(y);                    % all indices where y is not annotated
  s2hat = 1./sum(bsxfun(@rdivide,~indnany,lik.sigma2),2);
  y0 = y; y0(indnany)=0;
  yhat = s2hat.*sum(bsxfun(@rdivide,y0,lik.sigma2),2);
  
  g = {};
  j=0;
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})
      j=j+1;        
      l = ~indnany(:,j);                    % annotated entries for annotator j
      yg = zeros(size(yhat));
      yg(l) = (s2hat(l)/lik.sigma2(j)).*(yhat(l)-y(l,j));
      g{j} = yg;
    end
  end
end



function E = lik_mgaussian_e(lik, x, y)
%lik_mgaussian_e  Evaluate the likelihood dependent terms in the energy 
%                 function (un-normalized negative log marginal posterior)
%
%  Description
%    C = lik_mgaussian_e(LIK, TX, Y) takes in likelihood function
%    and input and multi-annotated output training data X, Y. Returns 
%    likelihood dependent terms in the energy function, i.e., 
%      -1/2 log det(hat(Sigma)) - sum_i sum_m~i log 1/sigma_m + 
%      +1/2 sum_i sum_m~i (y_i^m)^2/sigma_m^2 
%      - 1/2 sum_ihat(y_i)^2/hat(sigma)_i^2)

  indnany = isnan(y);                    % all indices where y is not annotated
  Nm = sum(~indnany);               % count number of annotations per annotator
  y0 = y; y0(indnany) = 0;                   % y0 is y with all nan values to 0
  sn2 = lik.sigma2;                                              % noise levels
  sn2hat = 1./sum(bsxfun(@rdivide,~indnany,sn2),2);          % reweighted sigma
  yhat = sn2hat.*sum(bsxfun(@rdivide,y0,sn2),2);          % reweighted output y
  
  E = -0.5*sum(log(sn2hat)) + Nm*log(sqrt(sn2))' ...
      + 0.5*sum(sum(bsxfun(@rdivide,y0.^2,sn2))) -0.5*sum(yhat.^2./sn2hat);  

end

function g = lik_mgaussian_eg(lik, x, y)
%lik_mgaussian_eg  Evaluate the gradient of the likelihood dependent terms 
%                  in the energy function (un-normalized negative log marginal 
%                  posterior)
%
%  Description
%    C = lik_mgaussian_e(LIK, TX, Y) takes in likelihood function
%    and input and multi-annotated output training data X, Y. Returns 
%    likelihood dependent terms in the energy function, i.e., 
%      -1/2 log det(hat(Sigma)) - sum_i sum_m~i log 1/sigma_m + 
%      +1/2 sum_i sum_m~i (y_i^m)^2/sigma_m^2 
%      - 1/2 sum_ihat(y_i)^2/hat(sigma)_i^2)

  indnany = isnan(y);                    % all indices where y is not annotated
  Nm = sum(~indnany);               % count number of annotations per annotator
  y0 = y; y0(indnany) = 0;                   % y0 is y with all nan values to 0
  sn2 = lik.sigma2;                                              % noise levels
  sn2hat = 1./sum(bsxfun(@rdivide,~indnany,sn2),2);          % reweighted sigma
  yhat = sn2hat.*sum(bsxfun(@rdivide,y0,sn2),2);          % reweighted output y
  
  g = [];
  j=0;
  for i=1:length(lik.sigma2)
    if ~isempty(lik.p.sigma2{i})
      j=j+1;        
      l = ~indnany(:,j);                    % annotated entries for annotator j
      g(j) = -0.5*sum(sn2hat(l)./sn2(j)) + Nm(j)/2 ...
             -0.5*(sum((y(l,j).^2)/sn2(j))) ...
             -0.5*sum((yhat(l).*(yhat(l)-2*y(l,j)))/sn2(j));              
    end
  end

end



function reclik = lik_mgaussian_recappend(reclik, ri, lik)
%RECAPPEND  Record append
%
%  Description
%    RECLIK = lik_mgaussian_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood function record structure RECLIK, record index RI
%    and likelihood function structure LIK with the current MCMC
%    samples of the parameters. Returns RECLIK which contains all
%    the old samples and the current samples from LIK .
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
    % Initialize the record
    reclik.type = 'lik_mgaussian';
    
    % Initialize the parameters
    reclik.sigma2 = []; 
    reclik.n = []; 
    
    % Set the function handles
    reclik.fh.pak = @lik_mgaussian_pak;
    reclik.fh.unpak = @lik_mgaussian_unpak;
    reclik.fh.lp = @lik_mgaussian_lp;
    reclik.fh.lpg = @lik_mgaussian_lpg;
    reclik.fh.cfg = @lik_mgaussian_cfg;
    reclik.fh.trcov  = @lik_mgaussian_trcov;
    reclik.fh.trvar  = @lik_mgaussian_trvar;
    reclik.fh.recappend = @lik_mgaussian_recappend;  
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
  else
    % Append to the record
    likp = lik.p;

    % record sigma2
    reclik.sigma2(ri,:)=lik.sigma2;
    if isfield(likp,'sigma2') && ~isempty(likp.sigma2)
      reclik.p.sigma2 = likp.sigma2.fh.recappend(reclik.p.sigma2, ri, likp.sigma2);
    end
    % record n if given
    if isfield(lik,'n') && ~isempty(lik.n)
      reclik.n(ri,:)=lik.n(:)';
    end
  end
end
