%RUN_REPRODUCTION  Faithful reproduction of the published GDLC experiments.
%
%   Group 24 reproduction study of:
%       D. Wang, T. Li, P. Deng, F. Zhang, W. Huang, P. Zhang, J. Liu,
%       "A Generalized Deep Learning Clustering Algorithm Based on
%        Non-Negative Matrix Factorization", ACM TKDD, 2023.
%   Original implementation: https://github.com/Code706/GDLC  (Code706)
%
%   This script does NOT replace run_GDLC.m -- the authors' original entry
%   point is left untouched and still works.  It runs the SAME algorithm in
%   its original configuration (fixed learning rate, generalized bias
%   enabled) across the three text/microarray datasets reported in the
%   paper, records per-round ACC / NMI / objective, measures wall-clock
%   runtime, computes two extra metrics (ARI, Purity), and writes CSV +
%   text + figure outputs.
%
%   Usage:   clear; clc; close all; run_reproduction
%
%   NOTE: every reproduced number written by this script comes from an
%   actual MATLAB execution.  Nothing is hardcoded.

clc;
fprintf('==========================================================\n');
fprintf(' GDLC REPRODUCTION STUDY -- Group 24\n');
fprintf(' Original code: https://github.com/Code706/GDLC\n');
fprintf('==========================================================\n\n');

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);
addpath(fullfile(thisDir,'metrics'));
addpath(fullfile(thisDir,'utils'));

resultsDir = fullfile(thisDir,'results');
figuresDir = fullfile(thisDir,'figures');
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
if ~exist(figuresDir,'dir'), mkdir(figuresDir); end

% ----------------------------------------------------------------------- %
% Dataset configurations -- published hyper-parameters (Section: parameter
% settings of the paper / the authors' released defaults).
% ----------------------------------------------------------------------- %
cfg = struct('name',{},'file',{},'params',{},'paperACC',{},'paperNMI',{},'paperRuntime',{});

cfg(1).name = 'BASEHOCK';
cfg(1).file = 'BASEHOCK';
cfg(1).params = struct('C',2,'eta',3.5e-3, ...
                       'alpha1',1e-1,'beta1',1e-1, ...
                       'alpha2',1e-1,'beta2',1e-1);
cfg(1).paperACC = 0.991;  cfg(1).paperNMI = 0.935;  cfg(1).paperRuntime = 23.80;

cfg(2).name = 'PCMAC';
cfg(2).file = 'PCMAC';
cfg(2).params = struct('C',2,'eta',5e-3, ...
                       'alpha1',5e-2,'beta1',5e-2, ...
                       'alpha2',5e-2,'beta2',5e-2);
cfg(2).paperACC = 1.000;  cfg(2).paperNMI = 1.000;  cfg(2).paperRuntime = 15.60;

cfg(3).name = 'SMK_CAN_187';
cfg(3).file = 'SMK_CAN_187';
cfg(3).params = struct('C',2,'eta',2e-1, ...
                       'alpha1',5e-1,'beta1',5e-1, ...
                       'alpha2',5e-1,'beta2',5e-1);
cfg(3).paperACC = 0.995;  cfg(3).paperNMI = 0.957;  cfg(3).paperRuntime = 9.20;

% Shared reproduction settings: original algorithm, unmodified.
commonParams = struct('rounds',10,'lrMode','fixed','useBias',true,'verbose',true);

nData = numel(cfg);
summary = struct([]);

for d = 1:nData
    name = cfg(d).name;
    fprintf('\n----------------------------------------------------------\n');
    fprintf(' Dataset %d/%d: %s\n', d, nData, name);
    fprintf('----------------------------------------------------------\n');

    % ---- Preprocessing, identical in style to the authors' run_GDLC.m --- %
    S = load(cfg(d).file);
    if ~isfield(S,'X') || ~isfield(S,'Y')
        error('run_reproduction:badDataset', ...
              '%s.mat does not contain the expected X / Y variables.', cfg(d).file);
    end
    fea = S.X;
    gnd = S.Y;

    fea    = im2double(fea);
    nClass = length(unique(gnd));
    fea    = NormalizeFea(fea);

    fprintf('  samples = %d, features = %d, classes = %d\n', ...
            size(fea,1), size(fea,2), nClass);
    p = cfg(d).params;
    fprintf('  C = %d, eta = %g, alpha1 = %g, beta1 = %g, alpha2 = %g, beta2 = %g\n', ...
            p.C, p.eta, p.alpha1, p.beta1, p.alpha2, p.beta2);

    % ---- Merge dataset params with the common reproduction settings ----- %
    params = p;
    fn = fieldnames(commonParams);
    for f = 1:numel(fn)
        params.(fn{f}) = commonParams.(fn{f});
    end

    % ---- Run ------------------------------------------------------------ %
    tRun = tic;
    [LABEL, ACC, NMI, obj_NMF, extra] = GDLC(fea', nClass, gnd, params);
    runtimeSec = toc(tRun);

    % ---- Extra metrics on the FINAL round ------------------------------- %
    finalLabels = extra.rawLabels;              % raw cluster ids
    ariVal      = adjusted_rand_index(gnd, finalLabels);
    purityVal   = clustering_purity(gnd, finalLabels);

    fprintf('\n  Reproduced ACC     : %.6f   (paper %.3f)\n', ACC(end),  cfg(d).paperACC);
    fprintf('  Reproduced NMI     : %.6f   (paper %.3f)\n', NMI(end),  cfg(d).paperNMI);
    fprintf('  ARI                : %.6f\n', ariVal);
    fprintf('  Purity             : %.6f\n', purityVal);
    fprintf('  Runtime            : %.4f s (paper %.2f s)\n', runtimeSec, cfg(d).paperRuntime);

    % ---- Per-round CSV -------------------------------------------------- %
    roundsFile = fullfile(resultsDir, [name '_rounds.csv']);
    fid = fopen(roundsFile,'w');
    fprintf(fid,'Round,ACC,NMI,Objective\n');
    for r = 1:numel(ACC)
        fprintf(fid,'%d,%.6f,%.6f,%.6f\n', r, ACC(r), NMI(r), obj_NMF(r));
    end
    fclose(fid);
    fprintf('  wrote %s\n', roundsFile);

    % ---- Collect summary row -------------------------------------------- %
    s = struct();
    s.Dataset          = name;
    s.Paper_ACC        = cfg(d).paperACC;
    s.Reproduced_ACC   = ACC(end);
    s.Paper_NMI        = cfg(d).paperNMI;
    s.Reproduced_NMI   = NMI(end);
    s.ARI              = ariVal;
    s.Purity           = purityVal;
    s.Paper_Runtime    = cfg(d).paperRuntime;
    s.Repro_Runtime    = runtimeSec;
    s.roundACC         = ACC;
    s.roundNMI         = NMI;
    s.objective        = obj_NMF;
    if isempty(summary), summary = s; else, summary(end+1) = s; end %#ok<SAGROW>

    % Keep the learned representation of BASEHOCK for the optional PCA plot.
    if strcmp(name,'BASEHOCK')
        BASEHOCK_extra = extra; %#ok<NASGU>
        BASEHOCK_gnd   = gnd;   %#ok<NASGU>
        save(fullfile(resultsDir,'BASEHOCK_repro_extra.mat'), ...
             'BASEHOCK_extra','BASEHOCK_gnd');
    end
end

% ----------------------------------------------------------------------- %
% Summary CSV
% ----------------------------------------------------------------------- %
summaryFile = fullfile(resultsDir,'reproduction_summary.csv');
fid = fopen(summaryFile,'w');
fprintf(fid,['Dataset,Paper_ACC,Reproduced_ACC,ACC_Absolute_Difference,' ...
             'ACC_Percent_Difference,Paper_NMI,Reproduced_NMI,' ...
             'NMI_Absolute_Difference,NMI_Percent_Difference,ARI,Purity,' ...
             'Paper_Runtime_sec,Reproduced_Runtime_sec\n']);
for d = 1:numel(summary)
    s = summary(d);
    accAbs = abs(s.Paper_ACC - s.Reproduced_ACC);
    accPct = accAbs / s.Paper_ACC * 100;
    nmiAbs = abs(s.Paper_NMI - s.Reproduced_NMI);
    nmiPct = nmiAbs / s.Paper_NMI * 100;
    fprintf(fid,'%s,%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.4f,%.4f\n', ...
        s.Dataset, s.Paper_ACC, s.Reproduced_ACC, accAbs, accPct, ...
        s.Paper_NMI, s.Reproduced_NMI, nmiAbs, nmiPct, ...
        s.ARI, s.Purity, s.Paper_Runtime, s.Repro_Runtime);
end
fclose(fid);
fprintf('\nwrote %s\n', summaryFile);

% ----------------------------------------------------------------------- %
% Readable text report
% ----------------------------------------------------------------------- %
reportFile = fullfile(resultsDir,'reproduction_report.txt');
fid = fopen(reportFile,'w');
fprintf(fid,'==========================================================\n');
fprintf(fid,' GDLC REPRODUCTION REPORT -- Group 24\n');
fprintf(fid,' Paper : A Generalized Deep Learning Clustering Algorithm\n');
fprintf(fid,'         Based on Non-Negative Matrix Factorization\n');
fprintf(fid,'         Wang, Li, Deng, Zhang, Huang, Zhang, Liu -- ACM TKDD 2023\n');
fprintf(fid,' Code  : https://github.com/Code706/GDLC (Code706)\n');
fprintf(fid,' Run at: %s\n', datestr(now));
fprintf(fid,'==========================================================\n\n');
fprintf(fid,'Configuration: original GDLC, fixed learning rate, generalized\n');
fprintf(fid,'bias enabled, %d rounds, fixed RNG seed rand(''twister'',5489).\n\n', commonParams.rounds);
for d = 1:numel(summary)
    s = summary(d);
    accAbs = abs(s.Paper_ACC - s.Reproduced_ACC);
    accPct = accAbs / s.Paper_ACC * 100;
    nmiAbs = abs(s.Paper_NMI - s.Reproduced_NMI);
    nmiPct = nmiAbs / s.Paper_NMI * 100;
    fprintf(fid,'----------------------------------------------------------\n');
    fprintf(fid,'DATASET: %s\n', s.Dataset);
    fprintf(fid,'----------------------------------------------------------\n');
    fprintf(fid,'  Metric      Paper       Reproduced   AbsDiff    %%Diff\n');
    fprintf(fid,'  ACC         %-11.3f %-12.6f %-10.6f %.4f\n', s.Paper_ACC, s.Reproduced_ACC, accAbs, accPct);
    fprintf(fid,'  NMI         %-11.3f %-12.6f %-10.6f %.4f\n', s.Paper_NMI, s.Reproduced_NMI, nmiAbs, nmiPct);
    fprintf(fid,'  Runtime(s)  %-11.2f %-12.4f\n', s.Paper_Runtime, s.Repro_Runtime);
    fprintf(fid,'\n  Additional metrics (this study, not reported in the paper):\n');
    fprintf(fid,'    ARI       %.6f\n', s.ARI);
    fprintf(fid,'    Purity    %.6f\n', s.Purity);
    fprintf(fid,'\n  Per-round progression:\n');
    fprintf(fid,'    Round   ACC        NMI        Objective\n');
    for r = 1:numel(s.roundACC)
        fprintf(fid,'    %-7d %-10.6f %-10.6f %.6f\n', r, s.roundACC(r), s.roundNMI(r), s.objective(r));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'Note: paper values are reference values transcribed from the\n');
fprintf(fid,'publication.  Reproduced values were produced by this script.\n');
fclose(fid);
fprintf('wrote %s\n', reportFile);

% ----------------------------------------------------------------------- %
% Figures
% ----------------------------------------------------------------------- %
names   = {summary.Dataset};
paperA  = [summary.Paper_ACC];      repA = [summary.Reproduced_ACC];
paperN  = [summary.Paper_NMI];      repN = [summary.Reproduced_NMI];
paperR  = [summary.Paper_Runtime];  repR = [summary.Repro_Runtime];

plot_grouped_bar(names, paperA, repA, 'ACC', ...
    'GDLC ACC: Paper vs Reproduced', fullfile(figuresDir,'acc_comparison.png'), [0 1.05]);
plot_grouped_bar(names, paperN, repN, 'NMI', ...
    'GDLC NMI: Paper vs Reproduced', fullfile(figuresDir,'nmi_comparison.png'), [0 1.05]);
plot_grouped_bar(names, paperR, repR, 'Runtime (s)', ...
    'GDLC Runtime: Paper vs Reproduced', fullfile(figuresDir,'runtime_comparison.png'), []);

for d = 1:numel(summary)
    s = summary(d);
    r = 1:numel(s.roundACC);

    f = figure('Visible','off','Color','w','Position',[100 100 800 500]);
    plot(r, s.roundACC, '-o', 'LineWidth', 2, 'MarkerSize', 7); hold on;
    plot(r, s.roundNMI, '-s', 'LineWidth', 2, 'MarkerSize', 7);
    grid on; box on;
    xlabel('GDLC Round','FontSize',13);
    ylabel('Score','FontSize',13);
    title(sprintf('%s -- Convergence (ACC / NMI)', s.Dataset), ...
          'FontSize',15,'FontWeight','bold','Interpreter','none');
    legend({'ACC','NMI'},'Location','southeast','FontSize',12);
    set(gca,'FontSize',12,'XTick',r); ylim([0 1.05]);
    saveFigurePNG(f, fullfile(figuresDir, [s.Dataset '_convergence.png']));

    f = figure('Visible','off','Color','w','Position',[100 100 800 500]);
    plot(r, s.objective, '-^', 'LineWidth', 2, 'MarkerSize', 7, 'Color', [0.85 0.33 0.10]);
    grid on; box on;
    xlabel('GDLC Round','FontSize',13);
    ylabel('Objective function value','FontSize',13);
    title(sprintf('%s -- Objective Function', s.Dataset), ...
          'FontSize',15,'FontWeight','bold','Interpreter','none');
    set(gca,'FontSize',12,'XTick',r);
    saveFigurePNG(f, fullfile(figuresDir, [s.Dataset '_objective.png']));
end

fprintf('\n==========================================================\n');
fprintf(' Reproduction finished.  See results/ and figures/.\n');
fprintf('==========================================================\n');
