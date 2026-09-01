%RUN_C_SENSITIVITY  Optional latent-dimension sensitivity study for GDLC.
%
%   Group 24 reproduction study.  Original algorithm: Code706/GDLC.
%
%   Runs BASEHOCK with C in {1, 2, 4, 8}, keeping the published BASEHOCK
%   eta / regularisation values, a fixed learning rate and the generalized
%   bias enabled.  C = 2 is the published setting, so that row doubles as a
%   consistency check against run_reproduction.
%
%   Usage:   clear; clc; close all; run_c_sensitivity
%
%   This experiment is OPTIONAL and independent of the main pipeline.

clc;
fprintf('==========================================================\n');
fprintf(' GDLC C-SENSITIVITY EXPERIMENT -- Group 24 (BASEHOCK)\n');
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
S      = load(datasetName);
fea    = im2double(S.X);
gnd    = S.Y;
nClass = length(unique(gnd));
fea    = NormalizeFea(fea);
Xt     = fea';

Cvalues = [1 2 4 8];
rows = zeros(numel(Cvalues), 6);   % C ACC NMI ARI Purity Runtime

for i = 1:numel(Cvalues)
    Cv = Cvalues(i);
    fprintf('\n----------------------------------------------------------\n');
    fprintf(' C = %d\n', Cv);
    fprintf('----------------------------------------------------------\n');

    params = struct('C',Cv,'eta',3.5e-3, ...
                    'alpha1',1e-1,'beta1',1e-1, ...
                    'alpha2',1e-1,'beta2',1e-1, ...
                    'rounds',10,'lrMode','fixed','useBias',true,'verbose',true);

    tRun = tic;
    [~, ACC, NMI, ~, extra] = GDLC(Xt, nClass, gnd, params);
    runtimeSec = toc(tRun);

    ariVal    = adjusted_rand_index(gnd, extra.rawLabels);
    purityVal = clustering_purity(gnd, extra.rawLabels);

    fprintf('\n  C=%d  ACC=%.6f  NMI=%.6f  ARI=%.6f  Purity=%.6f  Runtime=%.4f s\n', ...
            Cv, ACC(end), NMI(end), ariVal, purityVal, runtimeSec);

    rows(i,:) = [Cv, ACC(end), NMI(end), ariVal, purityVal, runtimeSec];
end

outFile = fullfile(extResultsDir,[datasetName '_C_sensitivity.csv']);
fid = fopen(outFile,'w');
fprintf(fid,'C,ACC,NMI,ARI,Purity,Runtime_sec\n');
for i = 1:size(rows,1)
    fprintf(fid,'%d,%.6f,%.6f,%.6f,%.6f,%.4f\n', rows(i,1), rows(i,2), ...
            rows(i,3), rows(i,4), rows(i,5), rows(i,6));
end
fclose(fid);
fprintf('\nwrote %s\n', outFile);

f = figure('Visible','off','Color','w','Position',[100 100 900 560]);
plot(rows(:,1), rows(:,2), '-o','LineWidth',2,'MarkerSize',8); hold on;
plot(rows(:,1), rows(:,3), '-s','LineWidth',2,'MarkerSize',8);
plot(rows(:,1), rows(:,4), '-^','LineWidth',2,'MarkerSize',8);
plot(rows(:,1), rows(:,5), '-d','LineWidth',2,'MarkerSize',8);
grid on; box on;
xlabel('Latent dimension C','FontSize',13);
ylabel('Score','FontSize',13);
title(sprintf('%s -- Sensitivity to Latent Dimension C', datasetName), ...
      'FontSize',15,'FontWeight','bold','Interpreter','none');
legend({'ACC','NMI','ARI','Purity'},'Location','best','FontSize',12);
set(gca,'FontSize',12,'XTick',Cvalues,'XScale','log');
ylim([0 1.05]);
saveFigurePNG(f, fullfile(extFiguresDir,[datasetName '_C_sensitivity.png']));

fprintf('\nC-sensitivity experiment finished.\n');
