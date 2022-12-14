function [g, gdata, gprior] = gp_g(w, gp, x, y, varargin)
%GP_G  Evaluate the gradient of energy (GP_E) for Gaussian Process
%
%  Description
%    G = GP_G(W, GP, X, Y, OPTIONS) takes a full GP parameter
%    vector W, GP structure GP, a matrix X of input vectors and a
%    matrix Y of target vectors, and evaluates the gradient G of
%    the energy function (gp_e). Each row of X corresponds to one
%    input vector and each row of Y corresponds to one target
%    vector.
%
%    [G, GDATA, GPRIOR] = GP_G(W, GP, X, Y, OPTIONS) also returns
%    separately the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GP_E, GP_PAK, GP_UNPAK, GPCF_*
%

% Copyright (c) 2007-2011 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2010 Heikki Peura

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

if isfield(gp,'latent_method') && ~strcmp(gp.latent_method,'MCMC')
  % use inference specific methods
  switch gp.latent_method
    case 'Laplace'
      switch gp.lik.type
        %case 'Softmax'
        %  fh_g=@gpla_softmax_g;
        %case {'Softmax2' 'Multinom'}
        %  fh_g=@gpla_mo_g;
        case {'Softmax' 'Multinom' 'Zinegbin' 'Coxph' 'Logitgp'}
          fh_g=@gpla_nd_g;
        otherwise
          fh_g=@gpla_g;
      end
    case 'EP'
      fh_g=@gpep_g;
  end
  switch nargout 
    case 1
      [g] = fh_g(w, gp, x, y, varargin{:});
    case 2
      [g, gdata] = fh_g(w, gp, x, y, varargin{:});
    case 3
      [g, gdata, gprior] = fh_g(w, gp, x, y, varargin{:});
  end
  return
end

ip=inputParser;
ip.FunctionName = 'GP_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))+isnan(x(:)))) % modified
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;
if ~all(isfinite(w(:)));
  % instead of stopping to error, return NaN
  g=NaN;
  gdata = NaN;
  gprior = NaN;
  return;
end

% unpak the parameters
gp=gp_unpak(gp, w);
ncf = length(gp.cf);
n=size(x,1);

g = [];
gdata = [];
gprior = [];

switch gp.type
  case 'FULL'
    % ============================================================
    % FULL
    % ============================================================
    % Evaluate covariance
    
    notpositivedefinite = 0;
    switch size(y,2)
      case 1
        % ============================================================
        % FULL - Single annotated output
        % ============================================================
        [K, C] = gp_trcov(gp,x);
        if issparse(C)
          % evaluate the sparse inverse
          invC = spinv(C);       
          [LD, notpositivedefinite] = ldlchol(C);
          if notpositivedefinite
            % instead of stopping to chol error, return NaN
            g=NaN;
            gdata = NaN;
            gprior = NaN;
            return;
          end
          if  (~isfield(gp,'meanf') && ~notpositivedefinite)
            b = ldlsolve(LD,y);        
          else
            [invNM invAt HinvC]=mean_gf(gp,x,C,LD,[],[],y,'gaussian');
          end
        else
          % evaluate the full inverse
          invC = inv(C);        
          if  ~isfield(gp,'meanf')          
            b = C\y;      
          else
            [invNM invAt HinvC]=mean_gf(gp,x,C,invC,[],[],y,'gaussian');
          end
        end

      otherwise
        % ============================================================
        % FULL - Multi-annotated output
        % ============================================================        
        [K, C] = gp_trcov(gp,x,[],y);
%         indnany = isnan(y);              % all indices where y is not annotated
%         Nm = sum(~indnany);         % count number of annotations per annotator
%         y0 = y; y0(indnany) = 0;         % y0 is y with all nan values set to 0
%         sn2 = gp.lik.sigma2;                             % assume lik_mgaussian
%         sn2hat = 1./sum(bsxfun(@rdivide,~indnany,sn2),2);    % reweighted sigma
%         yhat = sn2hat.*sum(bsxfun(@rdivide,y0,sn2),2);    % reweighted output y
        yhat = gp.lik.fh.y(gp.lik,x,y);
        
        if issparse(C)
          % evaluate the sparse inverse
          invC = spinv(C);       
          [LD, notpositivedefinite] = ldlchol(C);
          if notpositivedefinite
            % instead of stopping to chol error, return NaN
            g=NaN;
            gdata = NaN;
            gprior = NaN;
            return;
          end
          if  (~isfield(gp,'meanf') && ~notpositivedefinite)        
            b = ldlsolve(LD,yhat);
          else
            %[invNM invAt HinvC]=mean_gf(gp,x,C,LD,[],[],y,'gaussian');
          end
        else
          % evaluate the full inverse
          invC = inv(C);        
          if  ~isfield(gp,'meanf')        
            b = C\yhat;           % multi-annotated output, use reweighted output
          else
            %[invNM invAt HinvC]=mean_gf(gp,x,C,invC,[],[],y,'gaussian');
          end
        end    
    end



    
    
    

    % =================================================================
    % Gradient with respect to covariance function parameters
    if (~isempty(strfind(gp.infer_params, 'covariance')) && ~notpositivedefinite)
      for i=1:ncf
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        gpcf = gp.cf{i};        
        if ~(isfield(gp,'derivobs') && gp.derivobs)
          % No derivative observations
          DKff = gpcf.fh.cfg(gpcf, x);
          gprior_cf = -gpcf.fh.lpg(gpcf);
        else
          [n m]=size(x);
          %Case: input dimension is 1
          if m==1

            DKffa = gpcf.fh.cfg(gpcf, x);
            DKdf = gpcf.fh.cfdg(gpcf, x);
            DKdd = gpcf.fh.cfdg2(gpcf, x);
            gprior_cf = -gpcf.fh.lpg(gpcf);

            % DKff{1} -- d K / d magnSigma2
            % DKff{2} -- d K / d lengthScale
            DKff{1} = [DKffa{1}, DKdf{1}'; DKdf{1}, DKdd{1}];
            DKff{2} = [DKffa{2}, DKdf{2}'; DKdf{2}, DKdd{2}];
            
            %Case: input dimension is >1    
          else
            DKffa = gpcf.fh.cfg(gpcf, x);
            DKdf = gpcf.fh.cfdg(gpcf, x);
            DKdd = gpcf.fh.cfdg2(gpcf, x);
            gprior_cf = -gpcf.fh.lpg(gpcf);

            %Check whether ARD method is in use (with gpcf_sexp)
            Ard=length(gpcf.lengthScale);
            
            % DKff{1} - d K / d magnSigma2
            % DKff{2:end} - d K / d lengthScale(1:end)
            for i=1:2
              DKff{i}=[DKffa{i} DKdf{i}';DKdf{i} DKdd{i}];
            end
            
            %If ARD is in use
            if Ard>1
              for i=2+1:2+Ard-1
                DKff{i}=[DKffa{i} DKdf{i}';DKdf{i} DKdd{i}];
              end  
            end
          end
        end
        
        % Are there specified mean functions
        if  ~isfield(gp,'meanf')
          % Evaluate the gradient with respect to covariance function
          % parameters
          for i2 = 1:length(DKff)
            i1 = i1+1;  
            Bdl = b'*(DKff{i2}*b);
            Cdl = sum(sum(invC.*DKff{i2})); % help arguments
            gdata(i1)=0.5.*(Cdl - Bdl);
            gprior(i1) = gprior_cf(i2);
          end
        else 
            for i2 = 1:length(DKff)
                i1=i1+1;
                dA = -1*HinvC*DKff{i2}*HinvC';                  % d A / d th
                trA = sum(invAt(:).*dA(:));                 % d log(|A|) / dth
                dMNM = invNM'*(DKff{i2}*invNM);           % d M'*N*M / d th
                trK = sum(sum(invC.*DKff{i2}));       % d log(Ky�?�) / d th
                gdata(i1)=0.5*(-1*dMNM + trK + trA);
                gprior(i1) = gprior_cf(i2);
            end
        end
        
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end    
      end
    end
  
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov') && ~notpositivedefinite
      switch(size(y,2))
        case 1
          % ============================================================
          % FULL - Single annotated output
          % ============================================================
            
          % Evaluate the gradient from Gaussian likelihood
          DCff = gp.lik.fh.cfg(gp.lik, x);
          gprior_lik = -gp.lik.fh.lpg(gp.lik);
          for i2 = 1:length(DCff)
            i1 = i1+1;
            if ~isfield(gp,'meanf')
              if size(DCff{i2}) > 1
                yKy = b'*(DCff{i2}*b);
                trK = sum(sum(invC.*DCff{i2})); % help arguments
                gdata_zeromean(i1)=0.5.*(trK - yKy);
              else 
                yKy=DCff{i2}.*(b'*b);
                trK = DCff{i2}.*(trace(invC));
                gdata_zeromean(i1)=0.5.*(trK - yKy);
              end
              gdata(i1)=gdata_zeromean(i1);
            else
              if size(DCff{i2}) > 1
                trK = sum(sum(invC.*DCff{i2})); % help arguments
              else 
                trK = DCff{i2}.*(trace(invC));
              end
              dA = -1*HinvC*DCff{i2}*HinvC';                  % d A / d th
              trA = sum(invAt(:).*dA(:));                 % d log(|A|) / dth
              dMNM = invNM'*(DCff{i2}*invNM);           % d M'*N*M / d th
              gdata(i1)=0.5*(-1*dMNM + trA + trK);
            end
            gprior(i1) = gprior_lik(i2);
          end
          % Set the gradients of hyperparameter
          if length(gprior_lik) > length(DCff)
            for i2=length(DCff)+1:length(gprior_lik)
              i1 = i1+1;
              gdata(i1) = 0;
              gprior(i1) = gprior_lik(i2);
            end
          end    
        otherwise
          % ============================================================
          % FULL - Multi-annotated output
          % ============================================================        
          % Evaluate the gradient from Gaussian likelihood
          DCff = gp.lik.fh.cfg(gp.lik, x, [], y);
          gprior_lik = -gp.lik.fh.lpg(gp.lik);
          dyhat = gp.lik.fh.yg(gp.lik,x,y);
          gdata_lik = gp.lik.fh.eg(gp.lik,x,y);  % lik dependent gradients          
          for i2 = 1:length(DCff)
            i1 = i1+1;
            if ~isfield(gp,'meanf')
              if size(DCff{i2}) > 1
                yKy = b'*(DCff{i2}*b);
                trK = sum(sum(invC.*DCff{i2})); % help arguments
                gdata_zeromean(i1)=0.5.*(trK - yKy);
              else 
                yKy=DCff{i2}.*(b'*b);
                trK = DCff{i2}.*(trace(invC));
                gdata_zeromean(i1)=0.5.*(trK - yKy);
              end              
              gdata(i1)=gdata_zeromean(i1) + gdata_lik(i2) + b'*dyhat{i2};
            else
              if size(DCff{i2}) > 1
                trK = sum(sum(invC.*DCff{i2})); % help arguments
              else 
                trK = DCff{i2}.*(trace(invC));
              end
              dA = -1*HinvC*DCff{i2}*HinvC';                  % d A / d th
              trA = sum(invAt(:).*dA(:));                 % d log(|A|) / dth
              dMNM = invNM'*(DCff{i2}*invNM);           % d M'*N*M / d th
              gdata(i1)=0.5*(-1*dMNM + trA + trK) + gdata_lik(i2) + b'*dyhat{i2};
            end
            gprior(i1) = gprior_lik(i2);
          end
          % Set the gradients of hyperparameter
          if length(gprior_lik) > length(DCff)
            for i2=length(DCff)+1:length(gprior_lik)
              i1 = i1+1;
              gdata(i1) = 0;
              gprior(i1) = gprior_lik(i2);
            end
          end    
       end
    end  
    
    
    if ~isempty(strfind(gp.infer_params, 'mean')) && isfield(gp,'meanf') && ~notpositivedefinite
        
        nmf=numel(gp.meanf);
        [H,b,B]=mean_prep(gp,x,[]);
        M = H'*b-y;
        
        if issparse(C)
            [LD, notpositivedefinite] = ldlchol(C);
            if ~notpositivedefinite
                KH = ldlsolve(LD, H');
            end
            [LB, notpositivedefinite2] = chol(B);
            if ~notpositivedefinite2
                A = LB\(LB'\eye(size(B))) + H*KH;
                LA = chol(A);
                a = ldlsolve(LD, M) - KH*(LA\(LA'\(KH'*M)));
                iNH = ldlsolve(LD, H') - KH*(LA\(LA'\(KH'*H')));
            end
        else
            N = C + H'*B*H;
            [LN, notpositivedefinite3] = chol(N);
            if ~notpositivedefinite3
                a = LN\(LN'\M);
                iNH = LN\(LN'\H');
            end
        end
        if (~notpositivedefinite && ~notpositivedefinite2 && ~notpositivedefinite3)
            Ha=H*a;
            g_bb = (-H*a)';     % b and B parameters are log transformed in packing 
            indB = find(B>0);
            for i=1:length(indB)
                Bt = zeros(size(B)); Bt(indB(i))=1;
                BH = Bt*H;
                g_B(i) = 0.5* ( Ha'*Bt*Ha - sum(sum(iNH.*(BH'))) );
            end
            g_BB = g_B.*B(indB)';
            for i=1:nmf
                gpmf = gp.meanf{i};
                [lpg_b, lpg_B] = gpmf.fh.lpg(gpmf);
                ll=length(lpg_b);
                gdata = [gdata -g_bb((i-1)*ll+1:i*ll)];
                gprior = [gprior -lpg_b];
                ll=length(lpg_B);
                gdata = [gdata -g_B((i-1)*ll+1:i*ll)];
                gprior = [gprior -lpg_B];
            end
        else
          g=NaN;
          gdata = NaN;
          gprior = NaN;
          return
        end
    end
    
    g = gdata + gprior;
    
  case 'FIC'
    % ============================================================
    % FIC
    % ============================================================
    g_ind = zeros(1,numel(gp.X_u));
    gdata_ind = zeros(1,numel(gp.X_u));
    gprior_ind = zeros(1,numel(gp.X_u));

    u = gp.X_u;
    DKuu_u = 0;
    DKuf_u = 0;

    % First evaluate the needed covariance matrices
    % v defines that parameter is a vector
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu,'lower');
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements
                         % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:n
      iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    % ... then evaluate some help matrices.
    % A = K_uu+K_uf*inv(La)*K_fu
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;               % Ensure symmetry
    A = chol(A,'upper');
    L = iLaKfu/A;
    b = y'./Lav' - (y'*L)*L';
    iKuuKuf = Luu'\(Luu\K_fu');
    La = Lav;
    LL = sum(L.*L,2);
    
    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))
      % Loop over the covariance functions
      for i=1:ncf            
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % Get the gradients of the covariance matrices 
        % and gprior from gpcf_* structures
        gpcf = gp.cf{i};
        DKff = gpcf.fh.cfg(gpcf, x, [], 1);
        DKuu = gpcf.fh.cfg(gpcf, u); 
        DKuf = gpcf.fh.cfg(gpcf, u, x); 
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:length(DKuu)
          i1 = i1+1;       
          
          KfuiKuuKuu = iKuuKuf'*DKuu{i2};
          gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                             sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
          
          gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
          gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
          gdata(i1) = gdata(i1) + 0.5.*(sum(DKff{i2}./La) - sum(LL.*DKff{i2}));
          gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
          gprior(i1) = gprior_cf(i2);
        end
        
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
    end

    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= -0.5*DCff{i2}.*b*b';
        gdata(i1)= gdata(i1) + 0.5*sum(DCff{i2}./La-sum(L.*L,2).*DCff{i2});
        gprior(i1) = gprior_lik(i2);
      end
      % Set the gradients of hyperparameter
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end               
    end

    % =================================================================
    % Gradient with respect to inducing inputs
    if ~isempty(strfind(gp.infer_params, 'inducing'))
      if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
        m = size(gp.X_u,2);
        st=0;
        if ~isempty(gprior)
          st = length(gprior);
        end
        
        gdata(st+1:st+length(gp.X_u(:))) = 0;
        i1 = st+1;
        for i = 1:size(gp.X_u,1)
          if iscell(gp.p.X_u) % Own prior for each inducing input
            pr = gp.p.X_u{i};
            gprior(i1:i1+m) = -pr.fh.lpg(gp.X_u(i,:), pr);
          else % One prior for all inducing inputs
            gprior(i1:i1+m-1) = -gp.p.X_u.fh.lpg(gp.X_u(i,:), gp.p.X_u);
          end
          i1 = i1 + m;
        end
        
        % Loop over the covariance functions
        for i=1:ncf
          i1 = st;
          gpcf = gp.cf{i};
          DKuu = gpcf.fh.ginput(gpcf, u);
          DKuf = gpcf.fh.ginput(gpcf, u, x);
          
          for i2 = 1:length(DKuu)
            i1=i1+1;
            KfuiKuuKuu = iKuuKuf'*DKuu{i2};
            
            gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                          2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
            gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                          sum(LL.*sum(KfuiKuuKuu.*iKuuKuf',2)));
          end
        end
      end
    end
    
    g = gdata + gprior;
    
  case {'PIC' 'PIC_BLOCK'}
    % ============================================================
    % PIC
    % ============================================================
    g_ind = zeros(1,numel(gp.X_u));
    gdata_ind = zeros(1,numel(gp.X_u));
    gprior_ind = zeros(1,numel(gp.X_u));

    u = gp.X_u;
    ind = gp.tr_index;
    DKuu_u = 0;
    DKuf_u = 0;

    % First evaluate the needed covariance matrices
    % if they are not in the memory
    % v defines that parameter is a vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu,'lower');
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    %B=K_fu/Luu;
    B=Luu\K_fu';
    iLaKfu = zeros(size(K_fu));  % f x u
    for i=1:length(ind)
      Qbl_ff = B(:,ind{i})'*B(:,ind{i});
      [Kbl_ff, Cbl_ff] = gp_trcov(gp, x(ind{i},:));
      la = Cbl_ff - Qbl_ff;
      La{i} = (la + la')./2;
      iLaKfu(ind{i},:) = La{i}\K_fu(ind{i},:);
    end
    % ... then evaluate some help matrices.
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry

    L = iLaKfu/chol(A,'upper');
    b = zeros(1,n);
    b_apu=(y'*L)*L';
    for i=1:length(ind)
      b(ind{i}) = y(ind{i})'/La{i} - b_apu(ind{i});
    end
    iKuuKuf = Luu'\(Luu\K_fu');
    
    % =================================================================
    % Gradient with respect to covariance function parameters

    if ~isempty(strfind(gp.infer_params, 'covariance'))
      % Loop over the  covariance functions
      for i=1:ncf            
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % Get the gradients of the covariance matrices 
        % and gprior from gpcf_* structures
        gpcf = gp.cf{i};
        DKuu = gpcf.fh.cfg(gpcf, u); 
        DKuf = gpcf.fh.cfg(gpcf, u, x); 
        for kk = 1:length(ind)
          DKff{kk} = gpcf.fh.cfg(gpcf, x(ind{kk},:));
        end
        gprior_cf = -gpcf.fh.lpg(gpcf); 
        
        for i2 = 1:length(DKuu)
          i1 = i1+1;
          
          KfuiKuuKuu = iKuuKuf'*DKuu{i2};
          %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
          % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
          gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                             sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
          for kk=1:length(ind)
            gdata(i1) = gdata(i1) ...
                + 0.5.*(-b(ind{kk})*DKff{kk}{i2}*b(ind{kk})' ...
                        + 2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                        b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ... 
                        +trace(La{kk}\DKff{kk}{i2})...                                
                        - trace(L(ind{kk},:)*(L(ind{kk},:)'*DKff{kk}{i2})) ...
                        + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                        sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuKuu(ind{kk},:))*iKuuKuf(:,ind{kk})))));
          end
          gprior(i1) = gprior_cf(i2);
        end
        
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKuu)
          for i2=length(DKuu)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
    end
      
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= -0.5*DCff{i2}.*b*b';            
        for kk=1:length(ind)
          gdata(i1)= gdata(i1) + 0.5*trace((inv(La{kk})-L(ind{kk},:)*L(ind{kk},:)')).*DCff{i2};
        end
        gprior(i1) = gprior_lik(i2);
      end
      % Set the gradients of hyperparameter
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end
    end            
    
    % =================================================================
    % Gradient with respect to inducing inputs
    if ~isempty(strfind(gp.infer_params, 'inducing'))
      if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
        m = size(gp.X_u,2);
        
        st=0;
        if ~isempty(gprior)
          st = length(gprior);
        end
        gdata(st+1:st+length(gp.X_u(:))) = 0;
        
        i1 = st+1;
        for i = 1:size(gp.X_u,1)
          if iscell(gp.p.X_u) % Own prior for each inducing input
            pr = gp.p.X_u{i};
            gprior(i1:i1+m) = -pr.fh.lpg(gp.X_u(i,:), pr);
          else % One prior for all inducing inputs
            gprior(i1:i1+m-1) = -gp.p.X_u.fh.lpg(gp.X_u(i,:), gp.p.X_u);
          end
          i1 = i1 + m;
        end
        
        % Loop over the  covariance functions
        for i=1:ncf            
          i1=st;
          gpcf = gp.cf{i};
          DKuu = gpcf.fh.ginput(gpcf, u);
          DKuf = gpcf.fh.ginput(gpcf, u, x);
          
          for i2 = 1:length(DKuu)
            i1 = i1+1;
            KfuiKuuDKuu_u = iKuuKuf'*DKuu{i2};                
            gdata(i1) = gdata(i1) -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuDKuu_u))*(iKuuKuf*b') + 2.*sum(sum(L'.*((L'*DKuf{i2}')*iKuuKuf))) - ...
                                         sum(sum(L'.*((L'*KfuiKuuDKuu_u)*iKuuKuf))));
            
            for kk=1:length(ind)
              gdata(i1) = gdata(i1) + 0.5.*(2.*b(ind{kk})*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                                            b(ind{kk})*KfuiKuuDKuu_u(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})' ...
                                            + 2.*sum(sum(L(ind{kk},:)'.*(L(ind{kk},:)'*DKuf{i2}(:,ind{kk})'*iKuuKuf(:,ind{kk})))) - ...
                                            sum(sum(L(ind{kk},:)'.*((L(ind{kk},:)'*KfuiKuuDKuu_u(ind{kk},:))*iKuuKuf(:,ind{kk})))));
            end
          end
        end
      end
    end
    g = gdata + gprior;
    
  case 'CS+FIC'
    % ============================================================
    % CS+FIC
    % ============================================================
    g_ind = zeros(1,numel(gp.X_u));
    gdata_ind = zeros(1,numel(gp.X_u));
    gprior_ind = zeros(1,numel(gp.X_u));

    u = gp.X_u;
    DKuu_u = 0;
    DKuf_u = 0;

    cf_orig = gp.cf;

    cf1 = {};
    cf2 = {};
    j = 1;
    k = 1;
    for i = 1:ncf
      if ~isfield(gp.cf{i},'cs')
        cf1{j} = gp.cf{i};
        j = j + 1;
      else
        cf2{k} = gp.cf{i};
        k = k + 1;
      end
    end
    gp.cf = cf1;

    % First evaluate the needed covariance matrices
    % if they are not in the memory
    % v defines that parameter is a vector
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu,'lower');
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Qv_ff;   % 1 x f, Vector of diagonal elements

    gp.cf = cf2;
    K_cs = gp_trcov(gp,x);
    La = sparse(1:n,1:n,Lav,n,n) + K_cs;
    gp.cf = cf_orig;

    LD = ldlchol(La);
    %        iLaKfu = La\K_fu;
    iLaKfu = ldlsolve(LD, K_fu);

    % ... then evaluate some help matrices.
    % A = chol(K_uu+K_uf*inv(La)*K_fu))
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;            % Ensure symmetry
    L = iLaKfu/chol(A,'upper');
    %b = y'/La - (y'*L)*L';
    b = ldlsolve(LD,y)' - (y'*L)*L';
    
    siLa = spinv(La);
    idiagLa = diag(siLa);
    iKuuKuf = K_uu\K_fu';
    LL = sum(L.*L,2);
    
    % =================================================================
    % Gradient with respect to covariance function parameters
    if ~isempty(strfind(gp.infer_params, 'covariance'))
      % Loop over covariance functions 
      for i=1:ncf
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        gpcf = gp.cf{i};
        
        % Evaluate the gradient for FIC covariance functions
        if ~isfield(gpcf,'cs')
          % Get the gradients of the covariance matrices 
          % and gprior from gpcf_* structures
          DKff = gpcf.fh.cfg(gpcf, x, [], 1);
          DKuu = gpcf.fh.cfg(gpcf, u); 
          DKuf = gpcf.fh.cfg(gpcf, u, x); 
          gprior_cf = -gpcf.fh.lpg(gpcf);
          
          
          for i2 = 1:length(DKuu)
            i1 = i1+1;
            KfuiKuuKuu = iKuuKuf'*DKuu{i2};
            gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + 2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - ...
                               sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            
            temp1 = sum(KfuiKuuKuu.*iKuuKuf',2);
            temp2 = sum(DKuf{i2}'.*iKuuKuf',2);
            temp3 = 2.*DKuf{i2}' - KfuiKuuKuu;
            gdata(i1) = gdata(i1) - 0.5.*(b.*DKff{i2}')*b';
            gdata(i1) = gdata(i1) + 0.5.*(2.*b.*temp2'*b'- b.*temp1'*b');
            gdata(i1) = gdata(i1) + 0.5.*(sum(idiagLa.*DKff{i2} - LL.*DKff{i2}));   % corrected
            gdata(i1) = gdata(i1) + 0.5.*(2.*sum(LL.*temp2) - sum(LL.*temp1));
            
            %gdata(i1) = gdata(i1) + 0.5.*sum(sum(La\((2.*K_uf') - KfuiKuuKuu).*iKuuKuf',2));
            gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,temp3).*iKuuKuf',2));
            gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum(temp3.*iKuuKuf',2)) ); % corrected                
            gprior(i1) = gprior_cf(i2);                    
          end
          
          % Evaluate the gradient for compact support covariance functions
        else
          % Get the gradients of the covariance matrices 
          % and gprior from gpcf_* structures
          DKff = gpcf.fh.cfg(gpcf, x);
          gprior_cf = -gpcf.fh.lpg(gpcf);
          
          for i2 = 1:length(DKff)
            i1 = i1+1;
            gdata(i1) = 0.5*(sum(sum(siLa.*DKff{i2}',2)) - sum(sum(L.*(L'*DKff{i2}')')) - b*DKff{i2}*b');
            gprior(i1) = gprior_cf(i2);
          end
        end
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
    end
    
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= -0.5*DCff{i2}.*b*b';
        gdata(i1)= gdata(i1) + 0.5*sum(idiagLa-LL).*DCff{i2};
        gprior(i1) = gprior_lik(i2);
      end
      
      % Set the gradients of hyperparameter
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end
    end

    % =================================================================
    % Gradient with respect to inducing inputs
    if ~isempty(strfind(gp.infer_params, 'inducing'))
      if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
        m = size(gp.X_u,2);
        st=0;
        if ~isempty(gprior)
          st = length(gprior);
        end
        
        gdata(st+1:st+length(gp.X_u(:))) = 0;
        i1 = st+1;
        for i = 1:size(gp.X_u,1)
          if iscell(gp.p.X_u) % Own prior for each inducing input
            pr = gp.p.X_u{i};
            gprior(i1:i1+m) = -pr.fh.lpg(gp.X_u(i,:), pr);
          else % One prior for all inducing inputs
            gprior(i1:i1+m-1) = -gp.p.X_u.fh.lpg(gp.X_u(i,:), gp.p.X_u);
          end
          i1 = i1 + m;
        end
        
        for i=1:ncf
          i1=st;        
          gpcf = gp.cf{i};            
          if ~isfield(gpcf,'cs')
            DKuu = gpcf.fh.ginput(gpcf, u);
            DKuf = gpcf.fh.ginput(gpcf, u, x);
            
            
            for i2 = 1:length(DKuu)
              i1 = i1+1;
              KfuiKuuKuu = iKuuKuf'*DKuu{i2};
              
              gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                            2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
              gdata(i1) = gdata(i1) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
              gdata(i1) = gdata(i1) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                            sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));
              
              gdata(i1) = gdata(i1) + 0.5.*sum(sum(ldlsolve(LD,(2.*DKuf{i2}') - KfuiKuuKuu).*iKuuKuf',2));
              gdata(i1) = gdata(i1) - 0.5.*( idiagLa'*(sum((2.*DKuf{i2}' - KfuiKuuKuu).*iKuuKuf',2)) ); % corrected
            end
          end
        end
      end
    end
    
    g = gdata + gprior;
    
  case {'DTC' 'VAR' 'SOR'}
    % ============================================================
    % DTC/VAR/SOR
    % ============================================================
    g_ind = zeros(1,numel(gp.X_u));
    gdata_ind = zeros(1,numel(gp.X_u));
    gprior_ind = zeros(1,numel(gp.X_u));

    u = gp.X_u;
    DKuu_u = 0;
    DKuf_u = 0;

    % First evaluate the needed covariance matrices
    % v defines that parameter is a vector
    [Kv_ff, Cv_ff] = gp_trvar(gp, x);  % 1 x f  vector
    K_fu = gp_cov(gp, x, u);         % f x u
    K_uu = gp_trcov(gp, u);          % u x u, noiseles covariance K_uu
    K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
    Luu = chol(K_uu)';
    % Evaluate the Lambda (La)
    % Q_ff = K_fu*inv(K_uu)*K_fu'
    % Here we need only the diag(Q_ff), which is evaluated below
    B=Luu\(K_fu');
    Qv_ff=sum(B.^2)';
    Lav = Cv_ff-Kv_ff;   % 1 x f, Vector of diagonal elements
                         % iLaKfu = diag(inv(Lav))*K_fu = inv(La)*K_fu
    iLaKfu = zeros(size(K_fu));  % f x u,
    for i=1:n
      iLaKfu(i,:) = K_fu(i,:)./Lav(i);  % f x u
    end
    % ... then evaluate some help matrices.
    % A = K_uu+K_uf*inv(La)*K_fu
    A = K_uu+K_fu'*iLaKfu;
    A = (A+A')./2;               % Ensure symmetry
    A = chol(A);
    L = iLaKfu/A;
    b = y'./Lav' - (y'*L)*L';
    iKuuKuf = Luu'\(Luu\K_fu');
    La = Lav;
    LL = sum(L.*L,2);
    iLav=1./Lav;
    
    LL1=iLav-LL;
    
    % =================================================================
    
    if ~isempty(strfind(gp.infer_params, 'covariance'))
      % Loop over the covariance functions
      for i=1:ncf            
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % Get the gradients of the covariance matrices 
        % and gprior from gpcf_* structures
        gpcf = gp.cf{i};
        DKff = gpcf.fh.cfg(gpcf, x, [], 1);
        DKuu = gpcf.fh.cfg(gpcf, u); 
        DKuf = gpcf.fh.cfg(gpcf, u, x); 
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:length(DKuu)
          i1 = i1+1;       
          
          KfuiKuuKuu = iKuuKuf'*DKuu{i2};
          gdata(i1) = -0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
          gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                                        - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
          
          if strcmp(gp.type, 'VAR')
            gdata(i1) = gdata(i1) + 0.5.*(sum(iLav.*DKff{i2})-2.*sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2)) + ...
                                          sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))); % trace-term derivative
          end
          gprior(i1) = gprior_cf(i2);
        end
        
        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
    end      
    
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= -0.5*DCff{i2}.*b*b';
        gdata(i1)= gdata(i1) + 0.5*sum(DCff{i2}./La-sum(L.*L,2).*DCff{i2});
        if strcmp(gp.type, 'VAR')
          gdata(i1)= gdata(i1) + 0.5*(sum((Kv_ff-Qv_ff)./La));
        end
        
        gprior(i1) = gprior_lik(i2);                        
      end
      % Set the gradients of hyperparameter
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end               
    end        
    
    % =================================================================
    % Gradient with respect to inducing inputs
    if ~isempty(strfind(gp.infer_params, 'inducing'))
      if isfield(gp.p, 'X_u') && ~isempty(gp.p.X_u)
        m = size(gp.X_u,2);
        st=0;
        if ~isempty(gprior)
          st = length(gprior);
        end
        
        gdata(st+1:st+length(gp.X_u(:))) = 0;
        i1 = st+1;
        for i = 1:size(gp.X_u,1)
          if iscell(gp.p.X_u) % Own prior for each inducing input
            pr = gp.p.X_u{i};
            gprior(i1:i1+m) = -pr.fh.lpg(gp.X_u(i,:), pr);
          else % One prior for all inducing inputs
            gprior(i1:i1+m-1) = -gp.p.X_u.fh.lpg(gp.X_u(i,:), gp.p.X_u);
          end
          i1 = i1 + m;
        end
        
        % Loop over the covariance functions
        for i=1:ncf
          i1 = st;
          gpcf = gp.cf{i};
          DKuu = gpcf.fh.ginput(gpcf, u);
          DKuf = gpcf.fh.ginput(gpcf, u, x);
          
          for i2 = 1:length(DKuu)
            i1=i1+1;
            KfuiKuuKuu = iKuuKuf'*DKuu{i2};
            gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
            gdata(i1) = gdata(i1) + 0.5.*(2.*(sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2))-sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))))...
                                          - sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2))+ sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
            
            if strcmp(gp.type, 'VAR')
              gdata(i1) = gdata(i1) + 0.5.*(0-2.*sum(iLav'*sum(DKuf{i2}'.*iKuuKuf',2)) + ...
                                            sum(iLav'*sum(KfuiKuuKuu.*iKuuKuf',2)));
            end
          end
        end
      end
    end
    
    g = gdata + gprior;  
    
  case 'SSGP'        
    % ============================================================
    % SSGP
    % ============================================================
    % Predictions with sparse spectral sampling approximation for GP
    % The approximation is proposed by M. Lazaro-Gredilla, J. 
    % Quinonero-Candela and A. Figueiras-Vidal in Microsoft
    % Research technical report MSR-TR-2007-152 (November 2007)
    % NOTE! This does not work at the moment.

    % First evaluate the needed covariance matrices
    % v defines that parameter is a vector
    [Phi, S] = gp_trcov(gp, x);        % n x m and nxn sparse matrices
    Sv = diag(S);
    
    m = size(Phi,2);
    
    A = eye(m,m) + Phi'*(S\Phi);
    A = chol(A,'lower');
    L = (S\Phi)/A';

    b = y'./Sv' - (y'*L)*L';
    iSPhi = S\Phi;
    
    % =================================================================
    if ~isempty(strfind(gp.infer_params,'covariance'))
      % Loop over the covariance functions
      for i=1:ncf
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        gpcf = gp.cf{i};
        
        
        % Get the gradients of the covariance matrices 
        % and gprior from gpcf_* structures
        DKff = gpcf.fh.cfg(gpcf, x);
        gprior_cf = -gpcf.fh.lpg(gpcf);

        % Evaluate the gradient with respect to lengthScale
        for i2 = 1:length(DKff)
          i1 = i1+1;
          iSDPhi = S\DKff{i2};
          
          gdata(i1) = 0.5*( sum(sum(iSDPhi.*Phi,2)) + sum(sum(iSPhi.*DKff{i2},2)) );
          gdata(i1) = gdata(i1) - 0.5*( sum(sum(L'.*(L'*DKff{i2}*Phi' + L'*Phi*DKff{i2}'),1)) );
          gdata(i1) = gdata(i1) - 0.5*(b*DKff{i2}*Phi' + b*Phi*DKff{i2}')*b';
          gprior(i1) = gprior_cf(i2);
        end

        % Set the gradients of hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end        
      end
    end
    
    % =================================================================
    % Gradient with respect to Gaussian likelihood function parameters
    if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
      % Evaluate the gradient from Gaussian likelihood
      DCff = gp.lik.fh.cfg(gp.lik, x);
      gprior_lik = -gp.lik.fh.lpg(gp.lik);
      for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= -0.5*DCff{i2}.*b*b';
        gdata(i1)= gdata(i1) + 0.5*sum(1./Sv-sum(L.*L,2)).*DCff{i2};
        gprior(i1) = gprior_lik(i2);
      end
      
      % Set the gradients of hyperparameter                                
      if length(gprior_lik) > length(DCff)
        for i2=length(DCff)+1:length(gprior_lik)
          i1 = i1+1;
          gdata(i1) = 0;
          gprior(i1) = gprior_lik(i2);
        end
      end                
    end
    
    % =================================================================
    % Gradient with respect to inducing inputs
    if ~isempty(strfind(gp.infer_params, 'inducing'))
      for i=1:ncf
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        gpcf = gp.cf{i};
        
        gpcf.GPtype = gp.type;        
        [gprior_ind, DKuu, DKuf] = gpcf.fh.gind(gpcf, x, y, g_ind, gdata_ind, gprior_ind);
        
        for i2 = 1:length(DKuu)
          KfuiKuuKuu = iKuuKuf'*DKuu{i2};
          
          gdata_ind(i2) = gdata_ind(i2) - 0.5.*((2*b*DKuf{i2}'-(b*KfuiKuuKuu))*(iKuuKuf*b') + ...
                                                2.*sum(sum(L'.*(L'*DKuf{i2}'*iKuuKuf))) - sum(sum(L'.*((L'*KfuiKuuKuu)*iKuuKuf))));
          gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*b.*sum(DKuf{i2}'.*iKuuKuf',2)'*b'- b.*sum(KfuiKuuKuu.*iKuuKuf',2)'*b');
          gdata_ind(i2) = gdata_ind(i2) + 0.5.*(2.*sum(sum(L.*L,2).*sum(DKuf{i2}'.*iKuuKuf',2)) - ...
                                                sum(sum(L.*L,2).*sum(KfuiKuuKuu.*iKuuKuf',2)));                    
        end
      end
    end
    
    g = gdata + gprior;

end
