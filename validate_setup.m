%VALIDATE_SETUP  Pre-flight validation for the Group 24 GDLC reproduction.
%
%   Checks, in order:
%     1. required files / datasets are present and expose X and Y
%     2. toolbox dependencies (logsig, randsample, im2double)
%     3. every script and function parses (syntax check)
%     4. ARI and Purity unit tests
%     5. GDLC backward compatibility + fixed-mode equivalence on a small
%        synthetic problem (fast; the full BASEHOCK equivalence check is
%        done by comparing run_GDLC's output against run_reproduction's
%        BASEHOCK row)
%
%   Usage:   clear; clc; close all; validate_setup

clc;
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);
addpath(fullfile(thisDir,'metrics'));
addpath(fullfile(thisDir,'utils'));

fprintf('==========================================================\n');
fprintf(' VALIDATION -- GDLC reproduction study (Group 24)\n');
fprintf('==========================================================\n\n');

% ---- 1. Files ---------------------------------------------------------- %
fprintf('[1] Required files\n');
needed = {'GDLC.m','run_GDLC.m','NormalizeFea.m','litekmeans.m','bestMap.m', ...
          'MutualInfo.m','hungarian.m','logsig.m','run_reproduction.m', ...
          'run_extension_experiments.m','metrics/adjusted_rand_index.m', ...
          'metrics/clustering_purity.m'};
for i = 1:numel(needed)
    ok = exist(fullfile(thisDir,needed{i}),'file') == 2;
    fprintf('    %-42s %s\n', needed{i}, tick(ok));
end

fprintf('\n[2] Datasets\n');
dsets = {'BASEHOCK','PCMAC','SMK_CAN_187'};
for i = 1:numel(dsets)
    fn = fullfile(thisDir,[dsets{i} '.mat']);
    if exist(fn,'file') ~= 2
        fprintf('    %-14s MISSING\n', dsets{i});
        continue;
    end
    info = whos('-file', fn);
    vars = {info.name};
    hasXY = any(strcmp(vars,'X')) && any(strcmp(vars,'Y'));
    fprintf('    %-14s vars = {%s}  X/Y present: %s\n', ...
            dsets{i}, strjoin(vars,', '), tick(hasXY));
end

% ---- 3. Dependencies --------------------------------------------------- %
fprintf('\n[3] Dependencies\n');
checkFun('logsig',     'local shim logsig.m provides it if the toolbox is absent');
checkFun('randsample', 'Statistics and Machine Learning Toolbox (used by litekmeans)');
checkFun('im2double',  'Image Processing Toolbox (used by run_GDLC preprocessing)');
checkFun('spdiags',    'core MATLAB (used by NormalizeFea)');

% ---- 4. Syntax --------------------------------------------------------- %
fprintf('\n[4] Syntax check (parse only, nothing executed)\n');
files = {'GDLC.m','run_GDLC.m','run_reproduction.m','run_extension_experiments.m', ...
         'run_c_sensitivity.m','plot_Q_pca.m','logsig.m', ...
         'metrics/adjusted_rand_index.m','metrics/clustering_purity.m', ...
         'metrics/test_metrics.m','utils/plot_grouped_bar.m', ...
         'utils/plot_convergence_compare.m','utils/saveFigurePNG.m'};
for i = 1:numel(files)
    fn = fullfile(thisDir,files{i});
    if exist(fn,'file') ~= 2
        fprintf('    %-42s MISSING\n', files{i});
        continue;
    end
    msgs = checkcode(fn, '-id');
    errs = msgs(arrayfun(@(m) any(strncmp(m.id,{'PARSE','SYNER','MDOTM'},5)), msgs));
    if isempty(errs)
        fprintf('    %-42s %s\n', files{i}, tick(true));
    else
        fprintf('    %-42s SYNTAX ISSUES:\n', files{i});
        for k = 1:numel(errs)
            fprintf('        line %d: %s\n', errs(k).line, errs(k).message);
        end
    end
end

% ---- 5. Metric unit tests ---------------------------------------------- %
fprintf('\n[5] Metric unit tests\n');
test_metrics;

% ---- 6. GDLC API / fixed-mode equivalence on a tiny synthetic problem --- %
fprintf('\n[6] GDLC API and fixed-mode equivalence (synthetic 40x30 problem)\n');
rand('twister',1);            % GDLC re-seeds internally, so this is safe
Xs  = rand(40,30);            % mFea = 40 features, nSmp = 30 samples
gs  = [ones(15,1); 2*ones(15,1)];   % one label per sample (nSmp = 30)

[L3, a3, n3, o3] = GDLC(Xs, 2, gs);                                  % old API
p = struct('lrMode','fixed','useBias',true,'verbose',false);
[L4, a4, n4, o4, ex4] = GDLC(Xs, 2, gs, p);                          % new API

dACC = max(abs(a3(:) - a4(:)));
dNMI = max(abs(n3(:) - n4(:)));
dOBJ = max(abs(o3(:) - o4(:)));
fprintf('    old-API call returns 4 outputs                 %s\n', tick(numel(a3)==10));
fprintf('    new-API call returns 5 outputs                 %s\n', tick(isstruct(ex4)));
fprintf('    max |ACC diff|  = %.3e                        %s\n', dACC, tick(dACC < 1e-12));
fprintf('    max |NMI diff|  = %.3e                        %s\n', dNMI, tick(dNMI < 1e-12));
fprintf('    max |obj diff|  = %.3e                        %s\n', dOBJ, tick(dOBJ < 1e-10));
fprintf('    labels identical                              %s\n', ...
        tick(isequal(L3{end}, L4{end})));
fprintf('    extra.learningRates constant in fixed mode    %s\n', ...
        tick(all(abs(ex4.learningRates - ex4.learningRates(1)) < eps)));

pd = p; pd.lrMode = 'decay'; pd.decay = 0.1;
[~,~,~,~,exd] = GDLC(Xs, 2, gs, pd);
expected = 0.0035 ./ (1 + 0.1*(1:10));
fprintf('    decay schedule matches eta0/(1+0.1t)          %s\n', ...
        tick(max(abs(exd.learningRates - expected)) < 1e-12));

pb = p; pb.useBias = false;
[~,~,~,~,exb] = GDLC(Xs, 2, gs, pb);
fprintf('    no-bias mode runs and zeroes the bias term    %s\n', ...
        tick(all(exb.objBias == 0)));

fprintf('\n==========================================================\n');
fprintf(' Validation complete.  Review any lines not marked OK.\n');
fprintf('==========================================================\n');

function s = tick(ok)
if ok, s = 'OK'; else, s = '**FAILED**'; end
end

function checkFun(name, note)
w = which(name);
if isempty(w)
    fprintf('    %-12s NOT FOUND  -- %s\n', name, note);
else
    fprintf('    %-12s found: %s\n', name, w);
end
end
