%RUN_EXTENSION_EXPERIMENTS  Group 24 experimental extensions to GDLC.
%
%   Reproduction study of:
%       Wang, Li, Deng, Zhang, Huang, Zhang, Liu,
%       "A Generalized Deep Learning Clustering Algorithm Based on
%        Non-Negative Matrix Factorization", ACM TKDD, 2023.
%   Original implementation: https://github.com/Code706/GDLC (Code706).
%   GDLC itself was NOT invented by us; the extensions below are ours.
%
%   Three variants are compared on BASEHOCK (fastest of the three datasets,
%   kept single-dataset deliberately so the script finishes quickly):
%
%     A. Original_FixedLR : lrMode = 'fixed',  useBias = true   (baseline)
%     B. AdaptiveLR       : lrMode = 'decay',  useBias = true
%                           eta_t = eta_0 / (1 + decay * round), decay = 0.1
%     C. NoBias           : lrMode = 'fixed',  useBias = false  (ablation)
%
%   All other hyper-parameters are the published BASEHOCK values.
%
%   Usage:   clear; clc; close all; run_extension_experiments

clc;
fprintf('==========================================================\n');
fprintf(' GDLC EXTENSION EXPERIMENTS -- Group 24  (BASEHOCK)\n');
fprintf('==========================================================\n\n');

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);
addpath(fullfile(thisDir,'metrics'));
addpath(fullfile(thisDir,'utils'));

extResultsDir = fullfile(thisDir,'results','extensions');
extFiguresDir = fullfile(thisDir,'figures','extensions');
if ~exist(extResultsDir,'dir'), mkdir(extResultsDir); end
if ~exist(extFiguresDir,'dir'), mkdir(extFiguresDir); end

datasetName = 'BASEHOCK';

% ---- Preprocessing, identical to the authors' run_GDLC.m ---------------- %
S      = load(datasetName);
fea    = S.X;
gnd    = S.Y;
fea    = im2double(fea);
nClass = length(unique(gnd));
fea    = NormalizeFea(fea);
Xt     = fea';

fprintf('Dataset %s: %d samples, %d features, %d classes\n\n', ...
        datasetName, size(fea,1), size(fea,2), nClass);

% ---- Published BASEHOCK hyper-parameters -------------------------------- %
base = struct('C',2,'eta',3.5e-3, ...
              'alpha1',1e-1,'beta1',1e-1, ...
              'alpha2',1e-1,'beta2',1e-1, ...
              'rounds',10,'verbose',true);

exps = struct('name',{},'params',{});

exps(1).name = 'Original_FixedLR';
exps(1).params = base;
exps(1).params.lrMode  = 'fixed';
exps(1).params.useBias = true;

exps(2).name = 'AdaptiveLR';
exps(2).params = base;
exps(2).params.lrMode  = 'decay';
exps(2).params.decay   = 0.1;
exps(2).params.useBias = true;

exps(3).name = 'NoBias';
exps(3).params = base;
exps(3).params.lrMode  = 'fixed';
exps(3).params.useBias = false;

res = struct([]);

for e = 1:numel(exps)
    fprintf('\n----------------------------------------------------------\n');
    fprintf(' Experiment %d/%d: %s\n', e, numel(exps), exps(e).name);
    fprintf('   lrMode = %s, useBias = %d\n', ...
            exps(e).params.lrMode, exps(e).params.useBias);
    fprintf('----------------------------------------------------------\n');

    tRun = tic;
    [~, ACC, NMI, obj_NMF, extra] = GDLC(Xt, nClass, gnd, exps(e).params);
    runtimeSec = toc(tRun);

    ariVal    = adjusted_rand_index(gnd, extra.rawLabels);
    purityVal = clustering_purity(gnd, extra.rawLabels);

    fprintf('\n  ACC     : %.6f\n', ACC(end));
    fprintf('  NMI     : %.6f\n', NMI(end));
    fprintf('  ARI     : %.6f\n', ariVal);
    fprintf('  Purity  : %.6f\n', purityVal);
    fprintf('  Runtime : %.4f s\n', runtimeSec);

    % Per-round CSV
    roundsFile = fullfile(extResultsDir, ...
                          sprintf('%s_%s_rounds.csv', datasetName, exps(e).name));
    fid = fopen(roundsFile,'w');
    fprintf(fid,'Round,ACC,NMI,Objective,LearningRate\n');
    for r = 1:numel(ACC)
        fprintf(fid,'%d,%.6f,%.6f,%.6f,%.10g\n', ...
                r, ACC(r), NMI(r), obj_NMF(r), extra.learningRates(r));
    end
    fclose(fid);
    fprintf('  wrote %s\n', roundsFile);

    s = struct('name',exps(e).name,'ACC',ACC(end),'NMI',NMI(end), ...
               'ARI',ariVal,'Purity',purityVal,'Runtime',runtimeSec, ...
               'roundACC',ACC,'roundNMI',NMI,'objective',obj_NMF, ...
               'lr',extra.learningRates);
    if isempty(res), res = s; else, res(end+1) = s; end %#ok<SAGROW>
end

% ----------------------------------------------------------------------- %
% Summary CSV
% ----------------------------------------------------------------------- %
summaryFile = fullfile(extResultsDir, [datasetName '_extension_results.csv']);
fid = fopen(summaryFile,'w');
fprintf(fid,'Experiment,ACC,NMI,ARI,Purity,Runtime_sec\n');
for e = 1:numel(res)
    fprintf(fid,'%s,%.6f,%.6f,%.6f,%.6f,%.4f\n', ...
            res(e).name, res(e).ACC, res(e).NMI, res(e).ARI, res(e).Purity, res(e).Runtime);
end
fclose(fid);
fprintf('\nwrote %s\n', summaryFile);

% ----------------------------------------------------------------------- %
% Figures
% ----------------------------------------------------------------------- %
labels = {res.name};

% 1. Metric comparison (ACC / NMI / ARI / Purity)
metricData = [ [res.ACC]', [res.NMI]', [res.ARI]', [res.Purity]' ];
f = figure('Visible','off','Color','w','Position',[100 100 950 540]);
h = bar(metricData,'grouped'); %#ok<NASGU>
grid on; box on;
set(gca,'XTickLabel',labels,'FontSize',12,'TickLabelInterpreter','none');
ylabel('Score','FontSize',13);
title(sprintf('%s -- GDLC Variant Comparison', datasetName), ...
      'FontSize',15,'FontWeight','bold','Interpreter','none');
legend({'ACC','NMI','ARI','Purity'},'Location','southoutside', ...
       'Orientation','horizontal','FontSize',12);
ylim([0 1.08]);
saveFigurePNG(f, fullfile(extFiguresDir,[datasetName '_method_comparison.png']));

% 2. Runtime comparison
f = figure('Visible','off','Color','w','Position',[100 100 850 520]);
b = bar([res.Runtime],'FaceColor',[0.35 0.60 0.35]);  %#ok<NASGU>
grid on; box on;
set(gca,'XTickLabel',labels,'FontSize',12,'TickLabelInterpreter','none');
ylabel('Runtime (seconds)','FontSize',13);
title(sprintf('%s -- Runtime by Variant', datasetName), ...
      'FontSize',15,'FontWeight','bold','Interpreter','none');
rt = [res.Runtime];
for e = 1:numel(rt)
    text(e, rt(e), sprintf('%.2f s', rt(e)), 'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom','FontSize',11,'FontWeight','bold');
end
ylim([0, max(rt)*1.20 + eps]);
saveFigurePNG(f, fullfile(extFiguresDir,[datasetName '_runtime_comparison.png']));

% 3. Learning-rate schedule (fixed vs decaying)
idxFixed = find(strcmp(labels,'Original_FixedLR'),1);
idxDecay = find(strcmp(labels,'AdaptiveLR'),1);
f = figure('Visible','off','Color','w','Position',[100 100 850 520]);
r = 1:numel(res(idxFixed).lr);
plot(r, res(idxFixed).lr, '-o','LineWidth',2,'MarkerSize',7); hold on;
plot(r, res(idxDecay).lr, '-s','LineWidth',2,'MarkerSize',7);
grid on; box on;
xlabel('GDLC Round','FontSize',13);
ylabel('Learning rate \eta','FontSize',13);
title('Learning-Rate Schedule: Fixed vs Adaptive Decay', ...
      'FontSize',15,'FontWeight','bold');
legend({'Fixed \eta_0','Adaptive \eta_0/(1+\lambda t), \lambda = 0.1'}, ...
       'Location','northeast','FontSize',12);
set(gca,'FontSize',12,'XTick',r);
saveFigurePNG(f, fullfile(extFiguresDir,'learning_rate_schedule.png'));

% 4/5. Convergence comparisons
plot_convergence_compare(res, 'roundACC', 'ACC', ...
    sprintf('%s -- ACC per Round by Variant', datasetName), ...
    fullfile(extFiguresDir,[datasetName '_ACC_convergence_compare.png']));
plot_convergence_compare(res, 'roundNMI', 'NMI', ...
    sprintf('%s -- NMI per Round by Variant', datasetName), ...
    fullfile(extFiguresDir,[datasetName '_NMI_convergence_compare.png']));

fprintf('\n==========================================================\n');
fprintf(' Extension experiments finished.\n');
fprintf(' Results : results/extensions/\n');
fprintf(' Figures : figures/extensions/\n');
fprintf('==========================================================\n');
