function [LABEL, acc, NMI, obj_NMF, extra] = GDLC(X, nClass, gnd, params)
% GDLC  Generalized Deep Learning Clustering based on Non-negative Matrix
%       Factorization.
%
%   Original implementation: https://github.com/Code706/GDLC  (Code706)
%   Paper: D. Wang, T. Li, P. Deng, F. Zhang, W. Huang, P. Zhang, J. Liu,
%          "A Generalized Deep Learning Clustering Algorithm Based on
%           Non-Negative Matrix Factorization", ACM TKDD, 2023.
%
%   [LABEL, acc, NMI, obj_NMF] = GDLC(X, nClass, gnd)
%       Original call signature.  Reproduces the released BASEHOCK
%       behaviour exactly (C = 2, eta = 0.0035, alpha1 = beta1 = 0.1,
%       alpha2 = beta2 = 0.1, 10 rounds, fixed learning rate, bias on).
%
%   [LABEL, acc, NMI, obj_NMF, extra] = GDLC(X, nClass, gnd, params)
%       Parameterised call used by the reproduction / extension runners.
%
%   params fields (all optional, defaults reproduce the original code):
%       .C        low-dimensional matrix dimension          (default 2)
%       .eta      base learning rate                        (default 0.0035)
%       .alpha1   bias-m regularisation weight              (default 0.1)
%       .alpha2   bias-n regularisation weight              (default 0.1)
%       .beta1    W regularisation weight                   (default 0.1)
%       .beta2    Q regularisation weight                   (default 0.1)
%       .rounds   number of GDLC rounds                     (default 10)
%       .lrMode   'fixed' | 'decay'                         (default 'fixed')
%       .decay    decay constant used when lrMode='decay'   (default 0.1)
%       .useBias  true | false, generalized bias ablation   (default true)
%       .verbose  true | false, per-round console printing  (default true)
%
%   REPRODUCTION-STUDY NOTE
%   -----------------------
%   With params omitted, or with lrMode='fixed' and useBias=true, every
%   mathematical expression below is byte-for-byte the original one; only
%   hardcoded constants were replaced by the corresponding params fields.
%   The two experimental variants are clearly marked inline.

% ---------------------------------------------------------------- defaults
if nargin < 4 || isempty(params)
    params = struct();
end
if ~isfield(params,'C'),       params.C       = 2;      end
if ~isfield(params,'eta'),     params.eta     = 0.0035; end
if ~isfield(params,'alpha1'),  params.alpha1  = 0.1;    end
if ~isfield(params,'beta1'),   params.beta1   = 0.1;    end
if ~isfield(params,'beta2'),   params.beta2   = 0.1;    end
if ~isfield(params,'alpha2'),  params.alpha2  = 0.1;    end
if ~isfield(params,'rounds'),  params.rounds  = 10;     end
if ~isfield(params,'lrMode'),  params.lrMode  = 'fixed';end
if ~isfield(params,'decay'),   params.decay   = 0.1;    end
if ~isfield(params,'useBias'), params.useBias = true;   end
if ~isfield(params,'verbose'), params.verbose = true;   end

% Low-dimensional matrix dimension
C = params.C;

% Number of  clusters
K = nClass;

rand('twister',5489);
% Obtain the number of rows and columns for the objective matrix


[mFea,nSmp]=size(X);

%initialization
% w1=rand(mFea,C);
% q1=rand(nSmp,C);
% m1=rand(mFea,1);
% n1=rand(nSmp,1);

w=logsig(rand(mFea,C));
q=logsig(rand(nSmp,C));
m=logsig(rand(mFea,1));
n=logsig(rand(nSmp,1));
% NOTE: m and n are always drawn so that the RNG consumption -- and hence
% the initial w and q -- are identical whether or not the bias is used.
% When params.useBias is false they are simply never read nor updated.

Round = 0;

% Parameter Setting
eta0   = params.eta;
alpha1 = params.alpha1;
beta1  = params.beta1;
beta2  = params.beta2;
alpha2 = params.alpha2;

useBias = params.useBias;

% Extension bookkeeping (does not affect the algorithm)
learningRates = zeros(1,params.rounds);

%initialization
while Round < params.rounds
Round = Round+1;

% ---- Learning-rate schedule ------------------------------------------- %
% 'fixed' : original behaviour, etaCurrent == eta0 for every round.
% 'decay' : extension, etaCurrent = eta0 / (1 + decay * Round).
% Only the SCALAR step size changes; the gradient expressions are untouched.
switch lower(params.lrMode)
    case 'fixed'
        etaCurrent = eta0;
    case 'decay'
        etaCurrent = eta0 / (1 + params.decay * Round);
    otherwise
        error('GDLC:badLrMode','params.lrMode must be ''fixed'' or ''decay''.');
end
learningRates(Round) = etaCurrent;
eta = etaCurrent;   % keep the original variable name in the update loop

% Nonlinear Constrained NMF(NNMF)
for i=1:mFea      % Traversing the objective matrix
    for j=1:nSmp  % Traversing the objective matrix
        sum1=0;
        
        for kk=1:C
          sum1=sum1+ 1/(1+exp(-w(i,kk)))*1/(1+exp(-q(j,kk)));
        end        
        if useBias
            temp= X(i,j)-sum1-1/(1+exp(-m(i)))-1/(1+exp(-n(j)));
        else
            % NO-BIAS ABLATION: the additive generalized-bias contribution
            % sigma(m_i) + sigma(n_j) is removed from the reconstruction, so
            % the model is X ~= sigma(W)*sigma(Q)'.
            temp= X(i,j)-sum1;
        end
        
        % Update bias
        if useBias
            C1=1/(1+exp(-m(i)));
            m(i)=m(i)-eta*(temp*(-1)*C1*(1-C1)+alpha1*C1*(1-C1)*C1);
            D=1/(1+exp(-n(j)));
            n(j)=n(j)-eta*(temp*(-1)*D*(1-D)+alpha2*D*(1-D)*D);
        end
        % (NO-BIAS ABLATION: m and n carry no gradient, so they are not
        %  updated at all -- this also saves the two SGD steps per element.)
        
        % Update the elements in the low-dimensional matrix
        % (unchanged in both modes; only `temp` above differs)
        for kk=1:C  
            A=1/(1+exp(-w(i,kk)));
            w(i,kk)=w(i,kk)-eta*(temp*(-1)*1/(1+exp(-q(j,kk)))*A*(1-A)+beta1*A*(1-A)*A);
        end
        
        for kk=1:C
            B=1/(1+exp(-q(j,kk)));
            q(j,kk)=q(j,kk)-eta*(temp*(-1)*1/(1+exp(-w(i,kk)))*B*(1-B)+beta2*B*(1-B)*B);
        end
    end 
end

% Generalized Deep Learning clustering (GDLC) updates all elements in w,q,m,n
% Element updates in NNMF are transformed into generalized weights and generalized biases.
w=logsig(w);% Eq.(23) 
q=logsig(q);% Eq.(24) 
if useBias
    m=logsig(m);% Eq.(25) 
    n=logsig(n);% Eq.(26) 
end
% (NO-BIAS ABLATION: Eq.(25)/(26) describe the generalized-bias transform;
%  with no bias in the model there is no bias state to transform.)

% Calculate the value of the objective function
Q=q; W=w; m1=m; n1=n; 
if useBias
    Ux=[W,m1];Vx=[Q,n1];
else
    % NO-BIAS ABLATION: the bias column is dropped from the factorisation,
    % and the alpha1/alpha2 bias-regularisation term vanishes from the
    % objective (there is nothing to regularise).
    Ux=W;Vx=Q;
end
dX = Ux*Vx'-X;
obj_NMF1(Round) = sum(sum(dX.^2));
obj_NMF2(Round) = beta1*sum(sum(W.^2))+beta2*sum(sum(Q.^2));
if useBias
    bojbias3(Round)=   alpha1*sum(m.^2)+alpha2*sum(n.^2);
else
    bojbias3(Round)=   0;
end
obj_NMF4(Round)=obj_NMF1(Round)+obj_NMF2(Round)+bojbias3(Round);
obj_NMF(Round)=sqrt(obj_NMF4(Round));


% Obtain clustering accuracy
rand('twister',5489);
label = litekmeans(Q,K,'Replicates',20);
idx22 = bestMap(gnd,label);
LABEL{Round} = idx22;
acc(Round)= length(find(gnd == idx22))/length(gnd);%accuracy
NMI(Round) = MutualInfo(gnd,label);
rawLabel{Round} = label;
if params.verbose
    disp(['Number of Rounds:' num2str(Round)])
    disp(['Object Function Value: ',num2str(obj_NMF(Round))]);
    disp(['Acc: ',num2str(acc(Round))]);
    disp(['NMI: ',num2str(NMI(Round))]);
    disp('-------------------------------------------------------');
end

end

% ------------------------------------------------------ extra output struct
% Purely additive: the first four outputs above are untouched.
if nargout > 4
    extra = struct();
    extra.Q             = Q;              % learned low-dimensional sample repr.
    extra.W             = W;              % learned low-dimensional feature repr.
    extra.m             = m;
    extra.n             = n;
    extra.finalLabels   = LABEL{end};     % best-mapped labels of the last round
    extra.rawLabels     = rawLabel{end};  % raw litekmeans labels of last round
    extra.roundACC      = acc;
    extra.roundNMI      = NMI;
    extra.objective     = obj_NMF;
    extra.objReconstruct= obj_NMF1;
    extra.objWQ         = obj_NMF2;
    extra.objBias       = bojbias3;
    extra.learningRates = learningRates;
    extra.params        = params;
    extra.runtimeMetadata = struct('mFea',mFea,'nSmp',nSmp, ...
                                   'nClass',K,'rounds',params.rounds, ...
                                   'lrMode',params.lrMode, ...
                                   'useBias',useBias, ...
                                   'timestamp',datestr(now));
end

end

%==========================================================================%
